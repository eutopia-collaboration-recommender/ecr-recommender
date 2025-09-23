import pickle

import torch
from box import Box
from sqlalchemy import Engine
from torch_geometric.data import Data

from util.heterogeneous.dataset import DatasetEuCoHT

from util.postgres import create_sqlalchemy_engine

NO_BOOTSTRAP = 10


def main(engine: Engine) -> None:
    # Build the homogeneous graph
    # No data manipulation for bootstrapping
    if NO_BOOTSTRAP is None:
        dataset: DatasetEuCoHT = DatasetEuCoHT(pg_engine=engine, num_train=0.7)
        dataset.build_heterogeneous_graph()
        dataset.close_engine()
        dataset_name = 'dataset_heterogeneous'

        with open(f'../data/{dataset_name}.pkl', 'wb') as output:
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
    # For bootstrapping uncertainty
    else:
        for bootstrap_id in range(NO_BOOTSTRAP):
            dataset_name = f'dataset_heterogeneous_b{bootstrap_id}'
            dataset: DatasetEuCoHT = DatasetEuCoHT(pg_engine=engine, bootstrap_id=bootstrap_id, num_train=0.7)
            dataset.build_heterogeneous_graph()
            dataset.close_engine()
            with open(f'../data/{dataset_name}.pkl', 'wb') as output:
                pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # Read settings from config file
    config: Box = Box.from_yaml(filename="../config.yaml")

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
    main(engine=engine)
