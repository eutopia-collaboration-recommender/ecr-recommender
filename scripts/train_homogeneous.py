import pandas as pd
import psycopg2
import torch
from box import Box
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data
from tqdm import tqdm

from util.homogeneous.data import (
    get_edge_attributes_co_authors,
    get_mapper_to_contiguous_ids,
    get_node_attributes_author,
    query_edges_co_authors,
    query_nodes_author, to_pyg_data
)
from util.homogeneous.model import evaluate, ModelEuCoHM, test, train
from util.postgres import create_connection


def build_homogeneous_graph(connection: psycopg2.extensions.connection) -> Data:
    author_df: pd.DataFrame = query_nodes_author(conn=connection)
    co_authored_df: pd.DataFrame = query_edges_co_authors(conn=connection)

    # Get the mapping to contiguous IDs
    author_id_map: dict
    author_sid_map: dict
    author_id_map, author_sid_map = get_mapper_to_contiguous_ids(node_df=author_df,
                                                                 node_id_column='author_sid')
    # Apply the mapping to the dataframes
    author_df['author_node_id'] = author_df['author_sid'].map(author_id_map)
    co_authored_df['author_node_id'] = co_authored_df['author_sid'].map(author_id_map)
    co_authored_df['co_author_node_id'] = co_authored_df['co_author_sid'].map(author_id_map)

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
    return data


def train_and_evaluate(model, data, optimizer, scheduler, model_config):
    results: list = list()
    for epoch in tqdm(range(1, model_config['num_epochs'] + 1)):

        # ------ Train
        train_loss: float = train(
            model=model,
            data=data,
            optimizer=optimizer
        )
        # ------ Test
        test_loss: float = test(
            model=model,
            data=data
        )
        scheduler.step(test_loss)
        # ------ Evaluate
        eval_result: dict = evaluate(
            k=model_config['num_recommendations'],
            model=model,
            data=data
        )

        # Save results
        epoch_result = {
            'epoch': epoch,
            'train_loss_bpr': train_loss,
            'test_loss_bpr': test_loss,
            'precision@k': eval_result['precision@k'].compute(),
            'recall@k': eval_result['recall@k'].compute(),
            'map@k': eval_result['map@k'].compute(),
            'mrr@k': eval_result['mrr@k'].compute(),
            'ndcg@k': eval_result['ndcg@k'].compute(),
            'hit_rate@k': eval_result['hit_rate@k'].compute()
        }
        results.append(epoch_result)

        # Log results
        if epoch % 10 == 0:
            # Log model performance
            formatted_str = ', '.join([f'{key}: {epoch_result[key]:.4f}' for key in epoch_result.keys()])
            print(formatted_str)
    # Return results
    return results


def main(connection: psycopg2.extensions.connection,
         model_config: dict) -> None:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    data: Data = build_homogeneous_graph(connection=connection)

    model = ModelEuCoHM(
        num_nodes=data.num_nodes,
        input_channels=data.num_features,
        hidden_channels=model_config['hidden_channels'],
        k=model_config['num_recommendations']
    ).to(device)

    # Transfer to device
    data = data.to(device)
    optimizer: Optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=model_config['learning_rate']
    )
    scheduler: LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    results: list = train_and_evaluate(
        model=model,
        data=data,
        optimizer=optimizer,
        scheduler=scheduler,
        model_config=model_config
    )

    # Save results
    results_df: pd.DataFrame = pd.DataFrame(results)
    results_df.to_csv('../results/results_EuCoHM.csv', index=False)

    # Save model
    torch.save(model.state_dict(), '../models/model_EuCoHM.pth')


if __name__ == '__main__':
    # Read settings from config file
    config: Box = Box.from_yaml(filename="../config.yaml")

    # Model configuration
    model_config: dict = dict(
        num_recommendations=10,
        num_train=0.8,
        learning_rate=1e-2,
        num_epochs=100,
        hidden_channels=64
    )
    # Connect to Postgres
    connection: psycopg2.extensions.connection = create_connection(
        username=config.POSTGRES.USERNAME,
        password=config.POSTGRES.PASSWORD,
        host=config.POSTGRES.HOST,
        port=config.POSTGRES.PORT,
        database=config.POSTGRES.DATABASE,
        schema=config.POSTGRES.BQ_SCHEMA
    )

    # Read data from Postgres
    main(connection=connection, model_config=model_config)
