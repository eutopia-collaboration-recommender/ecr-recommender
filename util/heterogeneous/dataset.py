import torch
import sqlalchemy
import numpy as np
import pandas as pd
import polars as pl
import torch_geometric.transforms as T

from sqlalchemy import Engine
from typing_extensions import Tuple
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import structured_negative_sampling

from util.postgres import query
from util.torch_geometric import get_mapper_to_contiguous_ids


def query_nodes_author(conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
    # Get all authors data and value metrics about their collaboration
    author_query: str = f"""
    SELECT author_id,
           dummy_feature
    FROM g_eucoht_node_author
    """
    print("Querying author nodes...")
    author_df: pd.DataFrame = query(conn=conn, query_str=author_query)

    return author_df


def query_nodes_article(engine: Engine, batch_size: int = 10000) -> pd.DataFrame:
    raw_connection = engine.raw_connection()
    print("Querying article nodes...")
    with raw_connection.cursor() as cur:
        article_query = f"""
        SELECT article_id,
               article_citation_normalized_count,
               is_eutopia_collaboration,
               collaboration_novelty_index,
               article_embedding
        FROM g_eucoht_node_article
        """
        cur.execute(article_query)
        # Initialize the Polars DataFrame
        article_df_polars: pl.DataFrame = pl.DataFrame()
        ix = 0
        # Fetch in chunks
        while True:
            rows = cur.fetchmany(size=batch_size)
            if not rows:
                break
            # Append the rows to the Polars DataFrame
            df_chunk = pl.DataFrame(rows, schema=[
                "article_id",
                "article_citation_normalized_count",
                "is_eutopia_collaboration",
                "collaboration_novelty_index",
                "article_embedding"
            ], orient="row")

            # Concatenate chunk with the master DataFrame
            article_df_polars = pl.concat([article_df_polars, df_chunk], how="vertical")
            print(f"Rows fetched {batch_size} for batch {ix}")
            ix += 1
    # Convert the Polars DataFrame to a Pandas DataFrame
    article_df = article_df_polars.to_pandas()
    # Close the cursor and the connection
    cur.close()
    raw_connection.close()

    return article_df


def query_edges_co_authors(conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
    # Get all edges between authors and co-authors
    coauthored_query = f"""
    SELECT author_id,
           co_author_id,
           time,
           1 + eutopia_collaboration_count AS weight
    FROM g_eucoht_edge_co_authors
    """
    print("Querying co-authorship edge data...")
    coauthored_df = query(conn=conn, query_str=coauthored_query)

    return coauthored_df


def query_edges_publishes(conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
    # Get all edges between authors and co-authors
    coauthored_query = f"""
    SELECT author_id,
           article_id,
           time
    FROM g_eucoht_edge_publishes
    """
    print("Querying publishing edge data...")
    coauthored_df = query(conn=conn, query_str=coauthored_query)

    return coauthored_df


def get_node_attributes_author(author_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sort author dataframe
    sorted_author_df = author_df.sort_values(by='author_node_id')
    # Exclude columns AUTHOR_ID, AUTHOR_NODE_ID
    author_x_columns = list(filter(lambda x: x not in ('author_id', 'author_node_id'), sorted_author_df.columns))
    # Convert types
    x_author = sorted_author_df[author_x_columns].astype('float32').values

    # Normalize X using std scaler
    x_author = StandardScaler().fit_transform(x_author)

    # Add node index
    unique_nodes = sorted_author_df['author_node_id'].unique()
    node_ids_author = torch.arange(len(unique_nodes))

    return x_author, node_ids_author


def get_node_attributes_article(article_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sort article dataframe
    sorted_article_df = article_df.sort_values(by='article_node_id')
    article_x_columns = list(
        filter(lambda x: x not in ('article_id', 'article_node_id', 'article_embedding'), sorted_article_df.columns))

    # Convert EMBEDDING_TENSOR_DATA to proper format
    embedding_tensor = np.stack(sorted_article_df['article_embedding'].values)
    embedding_tensor = torch.tensor(embedding_tensor, dtype=torch.float)

    # Convert types
    x_article = sorted_article_df[article_x_columns].astype('float32').values

    # Append embedding_tensor to article_x
    x_article = np.concatenate((x_article, embedding_tensor), axis=1)

    # Normalize X using std scaler
    x_article = StandardScaler().fit_transform(x_article)

    # Add node index
    unique_nodes = sorted_article_df['article_node_id'].unique()
    node_ids_article = torch.arange(len(unique_nodes))

    return x_article, node_ids_article


def get_edge_attributes_co_authors(co_authored_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # Add edge index: for edges corresponding to authors co-authoring articles (author to author connection)
    author_node_ids = torch.from_numpy(co_authored_df['author_node_id'].values)
    coauthor_node_ids = torch.from_numpy(co_authored_df['co_author_node_id'].values)
    edge_index_co_authors = torch.stack([author_node_ids, coauthor_node_ids], dim=0)
    edge_time_co_authors = torch.from_numpy(np.array(co_authored_df['time'].values.astype('int64')))

    return edge_index_co_authors, edge_time_co_authors


def get_edge_attributes_publishes(published_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    # Add edge index: for edges corresponding to authors publishing articles (author to article connection)
    author_node_ids = torch.from_numpy(published_df['author_node_id'].values)
    article_node_ids = torch.from_numpy(published_df['article_node_id'].values)
    edge_index_publishes = torch.stack([author_node_ids, article_node_ids], dim=0)
    edge_time_publishes = torch.from_numpy(np.array(published_df['time'].values.astype('int64')))

    return edge_index_publishes, edge_time_publishes


class DatasetEuCoHT:
    def __init__(self, pg_engine: Engine,
                 num_train: float = 0.8,
                 target_edge_type: tuple = ('author', 'co_authors', 'author'),
                 target_node_type: str = 'author'):
        self.engine: Engine = pg_engine
        self.num_train: float = num_train
        self.data: Data = Data()
        self.target_edge_type: tuple = target_edge_type
        self.target_node_type: str = target_node_type
        self.author_id_map: dict = dict()
        self.author_node_id_map: dict = dict()

    def build_homogeneous_graph(self) -> (Data, dict, dict):
        with self.engine.connect() as conn:
            co_authored_df: pd.DataFrame = query_edges_co_authors(conn=conn)
            author_df: pd.DataFrame = query_nodes_author(conn=conn)

            published_df: pd.DataFrame = query_edges_publishes(conn=conn)
            article_df: pd.DataFrame = query_nodes_article(engine=self.engine)

            # Get the mapping to contiguous IDs
            # For authors
            author_node_id_map: dict
            author_id_map: dict
            author_node_id_map, author_id_map = get_mapper_to_contiguous_ids(node_df=author_df,
                                                                             id_column='author_id')
            # For articles
            article_node_id_map: dict
            article_id_map: dict
            article_node_id_map, article_id_map = get_mapper_to_contiguous_ids(node_df=article_df,
                                                                               id_column='article_id')

            # Apply the mapping to the dataframes
            # For authors
            author_df['author_node_id'] = author_df['author_id'].map(author_node_id_map)
            co_authored_df['author_node_id'] = co_authored_df['author_id'].map(author_node_id_map)
            co_authored_df['co_author_node_id'] = co_authored_df['co_author_id'].map(author_node_id_map)
            published_df['author_node_id'] = published_df['author_id'].map(author_node_id_map)
            # For articles
            article_df['article_node_id'] = article_df['article_id'].map(article_id_map)
            published_df['article_node_id'] = published_df['article_id'].map(article_node_id_map)

            # Get node attributes
            x_author: torch.Tensor
            node_ids_author: torch.Tensor
            x_author, node_ids_author = get_node_attributes_author(author_df=author_df)
            x_article: torch.Tensor
            node_ids_article: torch.Tensor
            x_article, node_ids_article = get_node_attributes_article(article_df=article_df)
            # Get edge attributes
            edge_index_co_authors: torch.Tensor
            edge_attr_co_authors: torch.Tensor
            edge_time_co_authors: torch.Tensor
            edge_index_co_authors, edge_time_co_authors = get_edge_attributes_co_authors(
                co_authored_df=co_authored_df
            )
            edge_index_publishes: torch.Tensor
            edge_attr_publishes: torch.Tensor
            edge_time_publishes: torch.Tensor
            edge_index_publishes, edge_time_publishes = get_edge_attributes_publishes(
                published_df=published_df
            )

            # Create the PyG Data object
            data = self.to_pyg_data(x_author=x_author,
                                    node_ids_author=node_ids_author,
                                    x_article=x_article,
                                    node_ids_article=node_ids_article,
                                    edge_index_co_authors=edge_index_co_authors,
                                    edge_time_co_authors=edge_time_co_authors,
                                    edge_index_publishes=edge_index_publishes,
                                    edge_time_publishes=edge_time_publishes)

            # Set the dataset attributes
            self.data = data
            self.author_id_map = author_id_map
            self.author_node_id_map = author_node_id_map

            # Return the PyG Data object
            return data, author_node_id_map, author_id_map

    def split_to_train_and_test(self, data: HeteroData):
        edge_types = data.metadata()[1]

        time = data[self.target_edge_type].time
        perm = time.argsort()
        # Find minimum time in `co_authors` edge that still resides in test set
        test_min_time = time[perm[int(self.num_train * perm.numel()):]].min()

        for edge_type in edge_types:
            # Define the training dataset for all edge types
            time = data[edge_type].time
            data[edge_type].train_mask = time < test_min_time
            data[edge_type].train_edge_index = data[edge_type].edge_index[:, data[edge_type].train_mask]
            data[edge_type].train_edge_time = time[data[edge_type].train_mask]

        data.train_edge_index_dict = {edge_type: data[edge_type].train_edge_index for edge_type in edge_types}

        # Add test dataset for `co_authors` edge type
        data[self.target_edge_type].test_mask = data[self.target_edge_type].time >= test_min_time
        data[self.target_edge_type].test_pos_edge_index = data[self.target_edge_type].edge_index[:,
                                                          data[self.target_edge_type].test_mask]
        data[self.target_edge_type].test_edge_time = data[self.target_edge_type].time[
            data[self.target_edge_type].test_mask]

        # Negative sampling
        test_neg_edge_index_i, test_neg_edge_index_j, test_neg_edge_index_k = structured_negative_sampling(
            edge_index=data[self.target_edge_type].test_pos_edge_index,
            num_nodes=data[self.target_node_type].num_nodes)
        test_neg_edge_index = torch.stack([test_neg_edge_index_i, test_neg_edge_index_k], dim=0)
        # Add negative samples to test data
        data[self.target_edge_type].test_neg_edge_index = test_neg_edge_index
        return data

    def to_pyg_data(self,
                    x_author: torch.Tensor,
                    x_article: torch.Tensor,
                    node_ids_author: torch.Tensor,
                    node_ids_article: torch.Tensor,
                    edge_index_co_authors: torch.Tensor,
                    edge_time_co_authors: torch.Tensor,
                    edge_index_publishes: torch.Tensor,
                    edge_time_publishes: torch.Tensor) -> Data:
        # Initialize the PyG Data object
        data = HeteroData()

        # Save node indices:
        data["article"].node_id = torch.arange(len(node_ids_article))
        data["author"].node_id = torch.arange(len(node_ids_author))
        # Add edge 'published'
        data["author", "publishes", "article"].edge_index = edge_index_publishes
        data["author", "publishes", "article"].time = edge_time_publishes
        # Add edge 'co_authors'
        data[('author', 'co_authors', 'author')].edge_index = edge_index_co_authors
        data[('author', 'co_authors', 'author')].time = edge_time_co_authors

        # Set X for article
        data["article"].x = torch.from_numpy(x_article)
        # Set X for author
        data["author"].x = torch.from_numpy(x_author)

        # Reverse edges
        data = T.ToUndirected(reduce='min')(data)

        # Split to train and test
        data = self.split_to_train_and_test(data)

        return data

    def close_engine(self):
        self.engine.dispose()
        self.engine = None
