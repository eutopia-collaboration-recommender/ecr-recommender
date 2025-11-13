import pickle

import torch
from box import Box
from sqlalchemy import Engine
from torch_geometric.data import Data

from util.homogeneous.dataset import DatasetEuCoHM

from util.postgres import create_sqlalchemy_engine

NO_BOOTSTRAP = 10


def main(engine: Engine) -> None:
    # Build the homogeneous graph
    # No data manipulation for bootstrapping
    if NO_BOOTSTRAP is None:
        dataset: DatasetEuCoHM = DatasetEuCoHM(pg_engine=engine, num_train=0.7, use_periodical_embedding_decay=True, use_top_keywords=True)
        dataset.build_homogeneous_graph()
        dataset.close_engine()

        with open(f'../data/{dataset.get_dataset_name()}.pkl', 'wb') as output:
            pickle.dump(dataset, output, pickle.HIGHEST_PROTOCOL)
    # For bootstrapping uncertainty
    else:
        for bootstrap_id in range(NO_BOOTSTRAP):
            dataset: DatasetEuCoHM = DatasetEuCoHM(pg_engine=engine, bootstrap_id=bootstrap_id, num_train=0.7, use_periodical_embedding_decay=True, use_top_keywords=True)
            dataset.build_homogeneous_graph()
            dataset.close_engine()
            with open(f'../data/{dataset.get_dataset_name()}.pkl', 'wb') as output:
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
