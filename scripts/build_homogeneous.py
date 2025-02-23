import pickle

import torch
from box import Box
from sqlalchemy import Engine
from torch_geometric.data import Data

from util.homogeneous.dataset import DatasetEuCoHM

from util.postgres import create_sqlalchemy_engine


def main(engine: Engine) -> None:
    # Build the homogeneous graph
    data: Data
    author_node_id_map: dict
    author_id_map: dict
    dataset: DatasetEuCoHM = DatasetEuCoHM(pg_engine=engine, use_periodical_embedding_decay=True)
    dataset.build_homogeneous_graph()

    with open('../data/dataset_homogeneous_periodical.pkl', 'wb') as output:
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
