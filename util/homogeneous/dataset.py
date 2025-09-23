import datetime
import pickle

import numpy as np
import pandas as pd
import polars as pl
import sqlalchemy
import torch
import hashlib
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sqlalchemy import Engine
from sqlalchemy.exc import (OperationalError, ProgrammingError)
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling
from tqdm import tqdm
from typing_extensions import Tuple

from util.homogeneous.query import query_article_embeddings, query_author_keyword_embeddings, query_edges_co_authors, \
    query_nodes_author, \
    query_nodes_author_articles, \
    query_nth_time_percentile


def load_dataset(dataset_filepath: str = None, device: str = 'cpu') -> Tuple[Data, dict, dict]:
    # Add the object as a safe global to shut down warning
    torch.serialization.add_safe_globals([DatasetEuCoHM])
    # Open the dataset file and save it to variable
    with open(dataset_filepath, 'rb') as file:
        dataset: DatasetEuCoHM = pickle.load(file)

    data = dataset.data
    # Transfer to device
    data = data.to(device)
    author_id_map = dataset.author_id_map
    author_node_id_map = dataset.author_node_id_map
    return data, author_id_map, author_node_id_map


def assert_bidirectional_edges(edges: torch.Tensor) -> None:
    # Transpose edge_index to get a list of edges
    edges: torch.Tensor = edges.t()  # Shape: [num_edges, 2]

    # Create reverse edges by swapping columns
    reverse_edges: torch.Tensor = edges[:, [1, 0]]

    # Check if all reverse edges exist in the original edge list
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


def get_mapper_to_contiguous_ids(node_df: pd.DataFrame, id_column: str) -> Tuple[dict, dict]:
    # Get unique nodes
    unique_nodes = node_df[id_column].unique()
    # Create a mapping from node IDs to contiguous IDs
    node_id_map = {node: i for i, node in enumerate(unique_nodes)}
    # Create a mapping from contiguous IDs to node IDs
    id_map = {y: x for x, y in node_id_map.items()}

    # Return the mappings
    return node_id_map, id_map


def get_node_attributes_author(
        author_df: pd.DataFrame,
        use_top_keywords: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Sort author dataframe
    sorted_author_df = author_df.sort_values(by='author_node_id')
    # Exclude columns AUTHOR_ID, AUTHOR_NODE_ID
    excluded_columns = ['author_id', 'author_node_id', 'author_embedding', 'keyword_popularity_embedding']
    x_author_columns = list(
        filter(lambda x: x not in excluded_columns, sorted_author_df.columns))

    # Convert author_embedding to proper format
    embedding_tensor_author = np.stack(sorted_author_df['author_embedding'].values)
    embedding_tensor_author = torch.tensor(embedding_tensor_author, dtype=torch.float)

    # Convert types
    x_author = sorted_author_df[x_author_columns].astype('float64').values
    x_author = torch.tensor(x_author, dtype=torch.float)

    # Include top keywords to the author features
    if use_top_keywords:
        # Convert keyword_popularity_embedding to proper format
        embedding_tensor_keyword_popularity = np.stack(sorted_author_df['keyword_popularity_embedding'].values)
        # Perform PCA reduction on keyword popularity embeddings: 345 components explain 80% of the variance as per (`notebooks/Top keyword embedding reduction.ipynb`)
        pca = PCA(n_components=345)
        embedding_tensor_keyword_popularity = pca.fit_transform(embedding_tensor_keyword_popularity)
        embedding_tensor_keyword_popularity = torch.tensor(embedding_tensor_keyword_popularity, dtype=torch.float)

        # Append embedding tensors to x_author
        x_author = torch.cat((x_author, embedding_tensor_author, embedding_tensor_keyword_popularity), dim=1)
    else:
        # Append embedding tensors to x_author
        x_author = torch.cat((x_author, embedding_tensor_author), dim=1)

    # Normalize X using std scaler
    x_author = StandardScaler().fit_transform(x_author)

    # Add node index
    unique_nodes = sorted_author_df['author_node_id'].unique()
    node_ids_author = torch.arange(len(unique_nodes))

    return x_author, node_ids_author


def get_edge_attributes_co_authors(co_authored_df: pd.DataFrame) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Sort article dataframe
    co_authored_df = co_authored_df.sort_values(by='author_node_id')

    # Get edge attributes
    co_authors_edge_attr_columns = list(filter(lambda x: x not in (
        'author_id', 'co_author_id', 'author_node_id', 'co_author_node_id', 'time'),
                                               co_authored_df.columns))

    # Convert types
    edge_attr_co_authors = co_authored_df[co_authors_edge_attr_columns].astype('int64').values

    # Add edge index: for edges corresponding to authors co-authoring articles (author to author connection)
    author_node_ids = torch.from_numpy(co_authored_df['author_node_id'].values.astype('int64'))
    coauthor_node_ids = torch.from_numpy(co_authored_df['co_author_node_id'].values.astype('int64'))
    edge_index_co_authors = torch.stack([author_node_ids, coauthor_node_ids], dim=0)
    edge_time_co_authors = torch.from_numpy(np.array(co_authored_df['time'].values.astype('int64')))
    edge_weight_co_authors = torch.from_numpy(np.ceil(np.log(co_authored_df['weight']) + 1).values.astype('int64'))

    return edge_index_co_authors, edge_attr_co_authors, edge_time_co_authors, edge_weight_co_authors


class DatasetEuCoHM:
    def __init__(self, pg_engine: Engine,
                 num_train: float = 0.8,
                 bootstrap_id: int = None,
                 use_periodical_embedding_decay: bool = False,
                 use_top_keywords: bool = False):
        # Postgres engine
        self.engine: Engine = pg_engine

        # Number of training sample
        self.num_train: float = num_train

        # Boostrap identifier
        self.bootstrap_id = bootstrap_id

        # Use periodical decay for embeddings that calculates a weighted sum of article embeddings, where the most
        # recent articles have the highest weight. This way we get a stronger signal for the model to learn from
        # most recent research periods of authors.
        self.use_periodical_embedding_decay: bool = use_periodical_embedding_decay
        # Include top keywords to the author features
        self.use_top_keywords: bool = use_top_keywords

        # Table name dependent on the number of periodical embeddings and top keywords flag
        self.table_name: str = self.get_author_node_table_name()

        # Placeholder class attributes for the dataset
        self.data: Data = Data()
        self.author_id_map: dict = dict()
        self.author_node_id_map: dict = dict()

    def get_author_node_table_name(self):
        base_prefix = 'g_eucohm_node_author'
        postfix = ''
        if self.use_periodical_embedding_decay:
            postfix += f'_periodical_decay'
        else:
            postfix = '_base'

        return f'{base_prefix}{postfix}'

    def get_dataset_name(self):
        base_prefix = 'dataset_homogeneous'
        postfix = ''
        if self.use_periodical_embedding_decay:
            postfix += f'_periodical_decay'
        if self.use_top_keywords:
            postfix += '_top_keywords'
        if self.num_train == 1:
            postfix += '_full'
        if postfix == '':
            postfix = '_base'

        if self.bootstrap_id is not None:
            postfix += f'_b{self.bootstrap_id}'

        return f'{base_prefix}{postfix}'

    def fill_author_article_embedding(self, conn: sqlalchemy.engine.base.Connection,
                                      filter_dt: datetime.datetime) -> pd.DataFrame:
        author_articles_df: pl.DataFrame = query_nodes_author_articles(conn=conn, filter_dt=filter_dt)
        article_embedding_df: pl.DataFrame = query_article_embeddings(engine=self.engine, filter_dt=filter_dt)

        # Go through all the authors and average their embeddings
        print("Processing the author embeddings...")
        author_embeddings: list = list()
        for author_id in tqdm(author_articles_df['author_id'].unique()):
            # Get all articles for the author
            articles: pl.Series = author_articles_df \
                .filter(pl.col('author_id') == author_id)['article_id']
            # Get the embeddings for the articles
            embeddings: pl.Series = article_embedding_df \
                .filter(pl.col('article_id').is_in(articles)) \
                .sort(by='article_publication_dt', descending=False)['article_embedding']

            # Calculate weights for the embeddings based on the publication date
            embedding_weights: np.ndarray = 1 / (((
                                                          (filter_dt - article_embedding_df \
                                                           .filter(pl.col('article_id').is_in(articles)) \
                                                           .sort(by='article_publication_dt', descending=False
                                                                 )[
                                                              'article_publication_dt']).dt.total_days() / 365).ceil() + 1).log() + 1)
            # Min cap the weights
            min_weight = 0.05
            embedding_weights = np.maximum(embedding_weights, min_weight)
            # Normalize the weights
            embedding_weights = embedding_weights / embedding_weights.sum()

            # Convert to numpy array
            embedding_ndarray = np.array(embeddings.to_list())

            if self.use_periodical_embedding_decay:
                # Calculate a weighted sum of the embeddings based on the publication date, where the most recent articles have the highest weight
                weighted_avg_embeddings = np.average(embedding_ndarray, axis=0, weights=embedding_weights)
                pg_avg_embedding = '{' + ','.join(map(str, weighted_avg_embeddings.tolist())) + '}'
            else:
                # Average the embeddings
                avg_embedding = np.mean(embedding_ndarray, axis=0)
                pg_avg_embedding = '{' + ','.join(map(str, avg_embedding.tolist())) + '}'

            citations = author_articles_df \
                .filter(pl.col('author_id') == author_id)['article_citation_normalized_count'] \
                .median()
            collaboration_novelty_index = author_articles_df \
                .filter(pl.col('author_id') == author_id)['collaboration_novelty_index'] \
                .median()

            # Append the author embedding to the list
            author_embeddings.append(dict(
                author_id=author_id,
                author_embedding=pg_avg_embedding,
                publication_count=len(articles),
                article_citation_normalized_count=float(citations if citations is not None else 0),
                collaboration_novelty_index=float(
                    collaboration_novelty_index if collaboration_novelty_index is not None else 0),

            ))

        # Create a DataFrame from the dictionary
        author_embedding_df = pd.DataFrame(author_embeddings)

        # Include top keywords to the author features
        author_keyword_popularity_df = query_author_keyword_embeddings(conn=conn, filter_dt=filter_dt)
        author_keyword_popularity_df['keyword_popularity_embedding'] = \
            author_keyword_popularity_df['keyword_popularity_embedding'].apply(
                lambda x: str(x).replace('[', '{').replace(']', '}')
            )
        author_embedding_df = author_embedding_df.merge(author_keyword_popularity_df, on='author_id', how='left')

        # Write the DataFrame to the database
        print("Writing the author embeddings to the database...")
        author_embedding_df.to_sql(self.table_name, self.engine, if_exists='replace')

        return author_embedding_df

    def split_to_train_and_test(self, data, filter_dt: datetime.datetime, bootstrap_id: int = None):
        time: torch.Tensor = data.edge_time
        perm: torch.Tensor = time.argsort()
        filter_time = int(filter_dt.strftime('%Y%m%d'))
        # Get indices where time is less than filter time
        train_index = perm[time[perm] <= filter_time]
        test_index = perm[time[perm] > filter_time]



        # Edge index
        data.train_pos_edge_index = data.edge_index[:, train_index]
        data.train_pos_edge_weight = data.edge_weight[train_index]
        data.test_pos_edge_index = data.edge_index[:, test_index]
        data.test_pos_edge_weight = data.edge_weight[test_index]

        # Add negative samples to test
        neg_edge_index_i, neg_edge_index_j, neg_edge_index_k = structured_negative_sampling(
            edge_index=data.test_pos_edge_index,
            num_nodes=data.num_nodes
        )
        data.test_neg_edge_index = torch.stack([neg_edge_index_i, neg_edge_index_k], dim=0)

        return data

    def to_pyg_data(self,
                    x_author: torch.Tensor,
                    node_ids_author: torch.Tensor,
                    edge_index_co_authors: torch.Tensor,
                    edge_attr_co_authors: torch.Tensor,
                    edge_time_co_authors: torch.Tensor,
                    edge_weight_co_authors: torch.Tensor,
                    filter_dt: datetime.datetime) -> Data:
        # Initialize the PyG Data object
        data: Data = Data()

        # Save node indices:
        data.node_id = node_ids_author
        # Add edge 'co_authors'
        data.edge_index = edge_index_co_authors
        data.edge_attr = torch.from_numpy(edge_attr_co_authors).to(torch.float)
        data.edge_time = edge_time_co_authors
        data.edge_weight = edge_weight_co_authors

        # Set X for author nodes
        data.x = torch.from_numpy(x_author).to(torch.float)

        # Metadata about number of features and nodes
        data.num_features = data.x.shape[1]
        data.num_nodes = data.x.shape[0]

        # Split to train and test
        data = self.split_to_train_and_test(data=data, filter_dt=filter_dt)

        return data

    def build_homogeneous_graph(self) -> (Data, dict, dict):
        with  self.engine.connect() as conn:
            filter_dt = query_nth_time_percentile(conn=conn, percentile=self.num_train)
            co_authored_df: pd.DataFrame = query_edges_co_authors(conn=conn)
            try:
                author_df: pd.DataFrame = query_nodes_author(conn=conn, table_name=self.table_name)
            except ProgrammingError as e:
                # Initiate a new connection since the last one was interrupted
                with self.engine.connect() as conn_2:
                    print(f"Table {self.table_name} does not exist. Creating the table...")
                    self.fill_author_article_embedding(conn=conn_2, filter_dt=filter_dt)
                    author_df: pd.DataFrame = query_nodes_author(conn=conn_2, table_name=self.table_name)

            # Get the mapping to contiguous IDs
            author_node_id_map: dict
            author_id_map: dict
            author_node_id_map, author_id_map = get_mapper_to_contiguous_ids(node_df=author_df,
                                                                             id_column='author_id')
            # Apply the mapping to the dataframes
            author_df['author_node_id'] = author_df['author_id'].map(author_node_id_map)
            co_authored_df['author_node_id'] = co_authored_df['author_id'].map(author_node_id_map)
            co_authored_df['co_author_node_id'] = co_authored_df['co_author_id'].map(author_node_id_map)

            # Drop all rows with NaN values in author_node_id or co_author_node_id
            # These are the authors that first published after the filter date, i.e. if we picture
            # ourselves in the past, we would not know these authors existed yet.
            co_authored_df = co_authored_df.dropna(subset=['author_node_id', 'co_author_node_id'])

            # BOOTSTRAP: When creating bootstrapping datasets, we drop 10% of edges randomly
            if self.bootstrap_id is not None:
                combined = (
                        np.minimum(co_authored_df['author_node_id'], co_authored_df['co_author_node_id']).astype('int64').astype(str)
                        + '-' +
                        np.maximum(co_authored_df['author_node_id'], co_authored_df['co_author_node_id']).astype('int64').astype(str)
                        + f'#{self.bootstrap_id}'
                )

                # Hash each row with md5
                hashed = combined.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())
                # Get unique hashes
                unique_hashes = hashed.dropna().unique()

                # Randomly choose 10% of unique hashes
                n_remove = int(len(unique_hashes) * 0.10)
                hashes_to_remove = np.random.choice(unique_hashes, size=n_remove, replace=False)
                co_authored_df = co_authored_df[~hashed.isin(hashes_to_remove)].reset_index(drop=True)  # Keep ≈90% of data

            # Get node attributes
            x_author: torch.Tensor
            node_ids_author: torch.Tensor
            x_author, node_ids_author = get_node_attributes_author(
                author_df=author_df,
                use_top_keywords=self.use_top_keywords
            )

            # Get edge attributes
            edge_index_co_authors: torch.Tensor
            edge_attr_co_authors: torch.Tensor
            edge_time_co_authors: torch.Tensor
            edge_weight_co_authors: torch.Tensor
            edge_index_co_authors, edge_attr_co_authors, edge_time_co_authors, edge_weight_co_authors = \
                get_edge_attributes_co_authors(
                    co_authored_df=co_authored_df
                )

            # Create the PyG Data object
            data = self.to_pyg_data(x_author=x_author,
                                    node_ids_author=node_ids_author,
                                    edge_index_co_authors=edge_index_co_authors,
                                    edge_attr_co_authors=edge_attr_co_authors,
                                    edge_time_co_authors=edge_time_co_authors,
                                    edge_weight_co_authors=edge_weight_co_authors,
                                    filter_dt=filter_dt)

            # Set the dataset attributes
            self.data = data
            self.author_id_map = author_id_map
            self.author_node_id_map = author_node_id_map

            # Return the PyG Data object
            return data, author_node_id_map, author_id_map

    def close_engine(self):
        self.engine.dispose()
        self.engine = None
