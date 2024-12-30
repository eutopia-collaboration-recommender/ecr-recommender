import logging

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


class DatasetEuCoBase:
    def __init__(self, pg_engine: Engine, logger: logging.Logger):
        self.logger = logger
        self.engine: Engine = pg_engine
        self.table_name: str = 'g_eucohm_node_author'
        self.data: Data = None

    def query_nodes_author(self, conn: sqlalchemy.engine.base.Connection,
                           table_name: str) -> pd.DataFrame:
        # Get all authors data and value metrics about their collaboration
        author_query: str = f"""
        SELECT author_id,
               1 AS dummy_feature
        FROM g_included_author
        """
        self.logger.debug("Querying author nodes...")
        author_df: pd.DataFrame = query(conn=conn, query_str=author_query)
        self.logger.info(f"Number of author nodes found: {len(author_df)}")

        return author_df

    def query_edges_co_authors(self, conn: sqlalchemy.engine.base.Connection) -> pd.DataFrame:
        # Get all edges between authors and co-authors
        coauthored_query = f"""
        SELECT author_id,
               co_author_id,
               time
        FROM g_eucohm_edge_co_authors
        """
        self.logger.debug("Querying co-authorship edge data...")
        coauthored_df = query(conn=conn, query_str=coauthored_query)
        self.logger.info(f"Number of co-authorship edges found: {len(coauthored_df)}")
        return coauthored_df

    def build_homogeneous_graph(self) -> (Data, dict, dict):
        with  self.engine.connect() as conn:
            co_authored_df: pd.DataFrame = self.query_edges_co_authors(conn=conn)
            author_df: pd.DataFrame = self.query_nodes_author(conn=conn, table_name=self.table_name)

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
            # Save the data object to the class
            self.data = data
            # Return the PyG Data object
            return data, author_node_id_map, author_id_map


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
        filter(lambda x: x not in ('author_id', 'author_node_id'), sorted_author_df.columns))

    # Convert types
    x_author = sorted_author_df[x_author_columns].astype('float64').values
    # Convert to tensor
    x_author = torch.tensor(x_author, dtype=torch.float)
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
