import numpy as np
import pandas as pd
import polars as pl
import sqlalchemy
import torch

from sklearn.preprocessing import StandardScaler
from sqlalchemy import Engine
from sqlalchemy.exc import ProgrammingError
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from typing_extensions import Tuple
from util.postgres import query, query_polars


def query_nodes_author(conn: sqlalchemy.engine.base.Connection,
                       table_name: str) -> pd.DataFrame:
    # Get all authors data and value metrics about their collaboration
    author_query: str = f"""
    SELECT author_id,
           publication_count,
           author_embedding
    FROM {table_name}
    """
    print("Querying author nodes...")
    author_df: pd.DataFrame = query(conn=conn, query_str=author_query)

    return author_df


def query_article_embeddings(engine: Engine,
                             batch_size: int = 10000) -> pl.DataFrame:
    with engine.raw_connection().cursor() as cur:
        cur.execute("""
            SELECT article_id, article_embedding
            FROM g_included_article_embedding
        """)
        # Initialize the Polars DataFrame
        article_embedding_df: pl.DataFrame = pl.DataFrame()
        ix = 0
        # Fetch in chunks
        while True:
            rows = cur.fetchmany(size=batch_size)
            if not rows:
                break
            # Append the rows to the Polars DataFrame
            df_chunk = pl.DataFrame(rows, schema=["article_id", "article_embedding"], orient="row")

            # Concatenate chunk with the master DataFrame
            article_embedding_df = pl.concat([article_embedding_df, df_chunk], how="vertical")
            print(f"Rows fetched {batch_size} for batch {ix}")
            ix += 1

        return article_embedding_df


def query_nodes_author_articles(conn: sqlalchemy.engine.base.Connection) -> pl.DataFrame:
    # Get all authors data and value metrics about their collaboration
    author_articles_query: str = f"""
        SELECT author_id,
               article_id
        FROM g_eucohm_node_author_article
        """
    print("Querying author articles...")
    author_articles_df: pl.DataFrame = query_polars(conn=conn, query=author_articles_query)

    return author_articles_df


def query_edges_co_authors(conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
    # Get all edges between authors and co-authors
    coauthored_query = f"""
    SELECT author_id,
           co_author_id,
           time,
           1 + eutopia_collaboration_count AS weight
    FROM g_eucohm_edge_co_authors
    """
    print("Querying co-authorship edge data...")
    coauthored_df = query(conn=conn, query_str=coauthored_query)

    return coauthored_df


class DatasetEuCoHM:
    def __init__(self, pg_engine: Engine):
        self.engine: Engine = pg_engine
        self.table_name: str = 'g_eucohm_node_author'
        self.data: Data = None

    def build_homogeneous_graph(self) -> (Data, dict, dict):
        with  self.engine.connect() as conn:
            co_authored_df: pd.DataFrame = query_edges_co_authors(conn=conn)
            try:
                author_df: pd.DataFrame = query_nodes_author(conn=conn, table_name=self.table_name)
            except ProgrammingError as e:
                # Initiate a new connection since the last one was interrupted
                with self.engine.connect() as conn_2:
                    print(f"Table {self.table_name} does not exist. Creating the table...")
                    author_df: pd.DataFrame = self.fill_g_nodes_author_table(conn=conn_2)

            # Get the mapping to contiguous IDs
            author_node_id_map: dict
            author_id_map: dict
            author_node_id_map, author_id_map = get_mapper_to_contiguous_ids(node_df=author_df,
                                                                             id_column='author_id')
            # Apply the mapping to the dataframes
            author_df['author_node_id'] = author_df['author_id'].map(author_node_id_map)
            co_authored_df['author_node_id'] = co_authored_df['author_id'].map(author_node_id_map)
            co_authored_df['co_author_node_id'] = co_authored_df['co_author_id'].map(author_node_id_map)

            # Get node attributes
            x_author: torch.Tensor
            node_ids_author: torch.Tensor
            x_author, node_ids_author = get_node_attributes_author(author_df=author_df)
            # Get edge attributes
            edge_index_co_authors: torch.Tensor
            edge_attr_co_authors: torch.Tensor
            edge_time_co_authors: torch.Tensor
            edge_index_co_authors, edge_attr_co_authors, edge_time_co_authors = get_edge_attributes_co_authors(
                co_authored_df=co_authored_df)

            # Create the PyG Data object
            data = to_pyg_data(x_author=x_author,
                               node_ids_author=node_ids_author,
                               edge_index_co_authors=edge_index_co_authors,
                               edge_attr_co_authors=edge_attr_co_authors,
                               edge_time_co_authors=edge_time_co_authors)
            # Return the PyG Data object
            return data, author_node_id_map, author_id_map

    def fill_g_nodes_author_table(self, conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
        author_articles_df: pd.DataFrame = query_nodes_author_articles(conn=conn)
        article_embedding_df: pl.DataFrame = query_article_embeddings(engine=self.engine)

        # Go through all the authors and average their embeddings
        print("Processing the author embeddings...")
        author_embeddings: list = list()
        for author_id in tqdm(author_articles_df['author_id'].unique()):
            # Get all articles for the author
            articles: pd.Series

            articles: pl.Series = author_articles_df.filter(pl.col('author_id') == author_id)['article_id']
            # Get the embeddings for the articles
            embeddings: pl.Series = article_embedding_df.filter(pl.col('article_id').is_in(articles))[
                'article_embedding']
            embedding_values = embeddings.to_numpy()
            # Average the embeddings
            avg_embedding = np.mean(embedding_values, axis=0)
            pg_avg_embedding = '{' + ','.join(map(str, avg_embedding.tolist())) + '}'

            # Append the author embedding to the list
            author_embeddings.append(dict(
                author_id=author_id,
                author_embedding=pg_avg_embedding,
                publication_count=len(articles)
            ))

        # Create a DataFrame from the dictionary
        author_embedding_df = pd.DataFrame(author_embeddings)

        # Write the DataFrame to the database
        print("Writing the author embeddings to the database...")
        author_embedding_df.to_sql(self.table_name, self.engine, if_exists='replace')

        return author_embedding_df


def get_mapper_to_contiguous_ids(node_df: pd.DataFrame, id_column: str) -> Tuple[dict, dict]:
    # Get unique nodes
    unique_nodes = node_df[id_column].unique()
    # Create a mapping from node IDs to contiguous IDs
    node_id_map = {node: i for i, node in enumerate(unique_nodes)}
    # Create a mapping from contiguous IDs to node IDs
    id_map = {y: x for x, y in node_id_map.items()}

    # Return the mappings
    return node_id_map, id_map


def get_node_attributes_author(author_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sort author dataframe
    sorted_author_df = author_df.sort_values(by='author_node_id')
    # Exclude columns AUTHOR_ID, AUTHOR_NODE_ID
    x_author_columns = list(
        filter(lambda x: x not in ('author_id', 'author_node_id', 'author_embedding'), sorted_author_df.columns))

    # Convert EMBEDDING_TENSOR_DATA to proper format
    embedding_tensor_author = sorted_author_df['author_embedding'].apply(
        lambda x: np.array(x.replace('{', '').replace('}', '').split(',')).astype('float64'))
    embedding_tensor_author = torch.tensor(embedding_tensor_author, dtype=torch.float)

    # Convert types
    x_author = sorted_author_df[x_author_columns].astype('float64').values

    # Append embedding_tensor to x_author
    x_author = torch.tensor(x_author, dtype=torch.float)
    x_author = torch.cat((x_author, embedding_tensor_author), dim=1)

    # Normalize X using std scaler
    x_author = StandardScaler().fit_transform(x_author)

    # Add node index
    unique_nodes = sorted_author_df['author_node_id'].unique()
    node_ids_author = torch.arange(len(unique_nodes))

    return x_author, node_ids_author


def get_edge_attributes_co_authors(co_authored_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sort article dataframe
    co_authored_df = co_authored_df.sort_values(by='author_node_id')

    # Get edge attributes
    co_authors_edge_attr_columns = list(filter(lambda x: x not in (
        'author_id', 'co_author_id', 'author_node_id', 'co_author_node_id', 'time'),
                                               co_authored_df.columns))

    # Convert types
    edge_attr_co_authors = co_authored_df[co_authors_edge_attr_columns].astype('int64').values

    # Add edge index: for edges corresponding to authors co-authoring articles (author to author connection)
    author_node_ids = torch.from_numpy(co_authored_df['author_node_id'].values)
    coauthor_node_ids = torch.from_numpy(co_authored_df['co_author_node_id'].values)
    edge_index_co_authors = torch.stack([author_node_ids, coauthor_node_ids], dim=0)
    edge_time_co_authors = torch.from_numpy(np.array(co_authored_df['time'].values.astype('int64')))

    return edge_index_co_authors, edge_attr_co_authors, edge_time_co_authors


def split_to_train_and_test(data, num_train=0.8):
    time = data.edge_time
    perm = time.argsort()
    train_index = perm[:int(num_train * perm.numel())]
    test_index = perm[int(num_train * perm.numel()):]

    # Edge index
    data.train_pos_edge_index = data.edge_index[:, train_index]
    data.test_pos_edge_index = data.edge_index[:, test_index]

    # Add negative samples to test
    neg_edge_index_i, neg_edge_index_j, neg_edge_index_k = structured_negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes
    )
    data.test_neg_edge_index = torch.stack([neg_edge_index_i, neg_edge_index_k], dim=0)

    # data.edge_index = data.edge_attr = data.edge_time = None
    return data


def assert_bidirectional_edges(data: Data) -> None:
    # Step 1: Transpose edge_index to get a list of edges
    edges: torch.Tensor = data.edge_index.t()  # Shape: [num_edges, 2]

    # Step 2: Create reverse edges by swapping columns
    reverse_edges: torch.Tensor = edges[:, [1, 0]]

    # Step 3: Check if all reverse edges exist in the original edge list
    # Convert edges to a set of tuples for faster lookup
    edges_set: set = set(map(tuple, edges.tolist()))

    # Iterate over reverse edges and check for their presence
    missing_edges: list = list()
    for edge in reverse_edges.tolist():
        if tuple(edge) not in edges_set:
            missing_edges.append(edge)

    # If there are missing edges, raise an assertion error
    if missing_edges:
        raise AssertionError(f"The following edges are missing their reverse counterparts: {missing_edges}")


def to_pyg_data(x_author: torch.Tensor,
                node_ids_author: torch.Tensor,
                edge_index_co_authors: torch.Tensor,
                edge_attr_co_authors: torch.Tensor,
                edge_time_co_authors: torch.Tensor) -> Data:
    # Initialize the PyG Data object
    data: Data = Data()

    # Save node indices:
    data.node_id = node_ids_author
    # Add edge 'co_authors'
    data.edge_index = edge_index_co_authors
    data.edge_attr = torch.from_numpy(edge_attr_co_authors).to(torch.float)
    data.edge_time = edge_time_co_authors

    # Set X for author nodes
    data.x = torch.from_numpy(x_author).to(torch.float)

    # Metadata about number of features and nodes
    data.num_features = data.x.shape[1]
    data.num_nodes = data.x.shape[0]

    # Transform data to undirected graph
    # data = T.ToUndirected()(data)

    # Split to train and test
    data = split_to_train_and_test(data)

    # Test: check that the number of elements in the positive edge index equals to the number of elements in the negative edge index
    assert data.test_pos_edge_index.numel() == data.test_neg_edge_index.numel()
    # Test: check that the number of elements in the training, positive edge index equals to 0.8 times all nodes
    assert data.train_pos_edge_index.shape[1] == int(0.8 * data.edge_index.shape[1])
    # Test: check that the number of elements in the test, positive edge index equals to 0.2 times all nodes
    assert data.test_pos_edge_index.shape[1] == data.edge_index.shape[1] - int(0.8 * data.edge_index.shape[1])

    # Test: check that all edges are bidirectional
    assert_bidirectional_edges(data)

    return data
