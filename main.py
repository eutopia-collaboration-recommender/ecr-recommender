import torch

from box import Box
from fastapi import FastAPI
from pydantic import BaseModel
from torch_geometric.data import Data
from contextlib import asynccontextmanager

from util.homogeneous.dataset import DatasetEuCoHM
from util.homogeneous.model import ModelEuCoHM
from util.postgres import create_sqlalchemy_engine


# Define the API schema
class InferenceRequest(BaseModel):
    author_id: str


# Global variables for model
models: dict = dict()
datasets: dict = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI: Startup logic
    # Load configuration
    config: Box = Box.from_yaml(filename="config.yaml")

    # Connect to Postgres
    engine = create_sqlalchemy_engine(
        username=config.POSTGRES.USERNAME,
        password=config.POSTGRES.PASSWORD,
        host=config.POSTGRES.HOST,
        port=config.POSTGRES.PORT,
        database=config.POSTGRES.DATABASE,
        schema=config.POSTGRES.SCHEMA
    )

    # Build the homogeneous graph
    data: Data
    author_node_id_map: dict
    author_id_map: dict

    data, author_node_id_map, author_id_map = DatasetEuCoHM(pg_engine=engine).build_homogeneous_graph()
    datasets['EuCoHM'] = dict(
        data=data,
        author_node_id_map=author_node_id_map,
        author_id_map=author_id_map
    )

    # Initiate and load the model
    models['EuCoHM'] = ModelEuCoHM(
        input_channels=datasets['EuCoHM']['data'].num_features,
        hidden_channels=128,
        num_recommendations=10,
        num_layers=4,
        author_node_id_map=datasets['EuCoHM']['author_node_id_map'],
        author_id_map=datasets['EuCoHM']['author_id_map']
    )

    #  Load the model weights
    models['EuCoHM'].load_state_dict(torch.load('models/model_EuCoHM.pth', weights_only=True))
    models['EuCoHM'].eval()
    print("Model successfully loaded and initialized")

    # FastAPI: Return lifespan context
    yield

    # FastAPI: Shutdown logic
    if engine:
        engine.dispose()
        print("Database connection closed")


app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(request: InferenceRequest) -> list:
    # Use the preloaded model to make predictions
    model: ModelEuCoHM = models['EuCoHM']
    author_id: str = request.author_id
    print(f'Querying recommendations for author: {author_id}')

    recommendations: list = model.recommend(x=datasets['EuCoHM']['data'].x,
                                            edge_index=datasets['EuCoHM']['data'].train_pos_edge_index,
                                            author_id=author_id)
    return recommendations
