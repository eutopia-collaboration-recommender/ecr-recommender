import torch

from box import Box
from fastapi import FastAPI
from pydantic import BaseModel
from torch_geometric.data import Data
from contextlib import asynccontextmanager

from util.homogeneous.data import build_homogeneous_graph
from util.homogeneous.model import ModelEuCoHM
from util.postgres import create_connection


# Define the API schema
class InferenceRequest(BaseModel):
    author_sid: str


# Global variables for model
models: dict = dict()
datasets: dict = dict()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI: Startup logic
    # Load configuration
    config: Box = Box.from_yaml(filename="config.yaml")

    # Connect to Postgres
    connection = create_connection(
        username=config.POSTGRES.USERNAME,
        password=config.POSTGRES.PASSWORD,
        host=config.POSTGRES.HOST,
        port=config.POSTGRES.PORT,
        database=config.POSTGRES.DATABASE,
        schema=config.POSTGRES.BQ_SCHEMA
    )

    # Build the homogeneous graph
    data: Data
    author_id_map: dict
    author_sid_map: dict
    data, author_id_map, author_sid_map = build_homogeneous_graph(
        conn=connection
    )
    datasets['EuCoHM'] = dict(
        data=data,
        author_id_map=author_id_map,
        author_sid_map=author_sid_map
    )

    # Initiate and load the model
    models['EuCoHM'] = ModelEuCoHM(
        input_channels=datasets['EuCoHM']['data'].num_features,
        hidden_channels=config.MODEL.HOMOGENEOUS.HIDDEN_CHANNELS,
        k=config.MODEL.HOMOGENEOUS.NUM_RECOMMENDATIONS,
        author_id_map=datasets['EuCoHM']['author_id_map'],
        author_sid_map=datasets['EuCoHM']['author_sid_map']
    )

    #  Load the model weights
    models['EuCoHM'].load_state_dict(torch.load('models/model_EuCoHM.pth', weights_only=True))
    models['EuCoHM'].eval()
    print("Model successfully loaded and initialized")

    # FastAPI: Return lifespan context
    yield

    # FastAPI: Shutdown logic
    if connection:
        connection.close()
        print("Database connection closed")


app = FastAPI(lifespan=lifespan)


@app.post("/predict/")
async def predict(request: InferenceRequest) -> list:
    # Use the preloaded model to make predictions
    model: ModelEuCoHM = models['EuCoHM']
    author_sid: str = request.author_sid

    recommendations: list = model.recommend(x=datasets['EuCoHM']['data'].x,
                                            edge_index=datasets['EuCoHM']['data'].train_pos_edge_index,
                                            author_sid=author_sid)

    return recommendations
