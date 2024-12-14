from fastapi import FastAPI
from pydantic import BaseModel
import torch


# Define the API schema
class InferenceRequest(BaseModel):
    author_sid: str


app = FastAPI()


@app.post("/predict/")
async def predict(request: InferenceRequest):
    output = [1, 2, 3, 4, 5]
    return {
        'query': request.author_sid,
        'status': 'success',
        "predictions": output}
