from typing import Tuple

import pandas as pd
import psycopg2
import sqlalchemy
import torch
from box import Box
from sqlalchemy import Engine
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data
from tqdm import tqdm

from util.homogeneous.dataset import DatasetEuCoHM
from util.homogeneous.model import evaluate, ModelEuCoHM, test, train
from util.postgres import create_connection, create_sqlalchemy_engine


def train_and_evaluate(model: ModelEuCoHM,
                       data: Data,
                       optimizer: Optimizer,
                       scheduler: LRScheduler,
                       model_config: dict) -> list:
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
            'precision@k': float(eval_result['precision@k'].compute()),
            'recall@k': float(eval_result['recall@k'].compute()),
            'map@k': float(eval_result['map@k'].compute()),
            'mrr@k': float(eval_result['mrr@k'].compute()),
            'ndcg@k': float(eval_result['ndcg@k'].compute()),
            'hit_rate@k': float(eval_result['hit_rate@k'].compute())
        }
        results.append(epoch_result)

        # Log results
        if epoch % 10 == 0:
            # Log model performance
            formatted_str = ', '.join([f'{key}: {epoch_result[key]:.4f}' for key in epoch_result.keys()])
            print(formatted_str)

    # Return results
    return results


def main(engine: Engine,
         model_config: dict) -> None:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Build the homogeneous graph
    data: Data
    author_node_id_map: dict
    author_id_map: dict
    dataset: DatasetEuCoHM = DatasetEuCoHM(pg_engine=engine)
    data, author_node_id_map, author_id_map = dataset.build_homogeneous_graph()

    # Initialize the model
    model = ModelEuCoHM(
        input_channels=data.num_features,
        hidden_channels=model_config['hidden_channels'],
        k=model_config['num_recommendations'],
        author_node_id_map=author_node_id_map,
        author_id_map=author_id_map
    ).to(device)

    # Transfer to device
    data = data.to(device)

    # Initialize the optimizer
    optimizer: Optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=model_config['learning_rate']
    )

    # Initialize the scheduler
    scheduler: LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.5,
        patience=10
    )

    # Train and evaluate the model
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
        num_recommendations=config.MODEL.HOMOGENEOUS.NUM_RECOMMENDATIONS,
        num_train=config.MODEL.HOMOGENEOUS.NUM_TRAIN,
        learning_rate=config.MODEL.HOMOGENEOUS.LEARNING_RATE,
        num_epochs=config.MODEL.HOMOGENEOUS.NUM_EPOCHS,
        hidden_channels=config.MODEL.HOMOGENEOUS.HIDDEN_CHANNELS
    )
    # Connect to Postgres
    engine: Engine = create_sqlalchemy_engine(
        username=config.POSTGRES.USERNAME,
        password=config.POSTGRES.PASSWORD,
        host=config.POSTGRES.HOST,
        port=config.POSTGRES.PORT,
        database=config.POSTGRES.DATABASE,
        schema=config.POSTGRES.SCHEMA
    )

    # Read data from Postgres
    main(engine=engine, model_config=model_config)
