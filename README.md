# EUTOPIA Collaboration Recommender: Recommender System

**Author:** [@lukazontar](https://github.com/lukazontar)

<hr/>

## Project Introduction

This repository contains the code for the EUTOPIA collaboration recommender system.

In today's academic landscape, collaboration across disciplines and institutions is crucial due to complex scientific
papers. Yet, such collaboration is often underused, leading to isolated researchers and few connected hubs. This thesis
aims to create a system for proposing new partnerships based on research interests and trends, enhancing academic
cooperation. It focuses on building a network from scientific co-authorships and a recommender system for future
collaborations. Emphasis is on improving the EUTOPIA organization by fostering valuable, interdisciplinary academic
relationships.

The system consist of three main components:

1. **Luigi pipeline for data ingestion and
   transformation**: [ecr-luigi :octocat:](https://github.com/eutopia-collaboration-recommender/ecr-luigi).
2. **Analytical layer for gaining a deeper understanding of the
   data**: [ecr-analytics :octocat:](https://github.com/eutopia-collaboration-recommender/ecr-analytics).
3. **Recommender system for proposing new
   collaborations**: [ecr-recommender :octocat:](https://github.com/eutopia-collaboration-recommender/ecr-recommender).

<hr/>

## Setting up the environment

Environment stack:

- Python, SQL as main programming languages.
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/): a library for graph neural networks.
  *Installed via pip*.
- [FastAPI](https://fastapi.tiangolo.com/) for model inference via REST API. *Installed via pip*.

### Prerequisites

- Docker
- Python 3.10 (using [pyenv](https://github.com/pyenv-win/pyenv-win)
  and [venv](https://docs.python.org/3/library/venv.html))
- [Postgres](https://www.postgresql.org/) setup as the data warehouse.
  See [ecr-luigi](https://github.com/eutopia-collaboration-recommender/ecr-luigi) for more information.

To run Python scripts, you need to carefully set up the environment, specifically installing the PyTorch Geometric
library. We install the CPU version of both PyTorch and PyTorch Geometric.

First, we install PyTorch (CPU version):

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

If installation fails due to error: `Faiss assertion 'err == CUBLAS_STATUS_SUCCESS' failed` install `faiss`:

```
pip install https://github.com/kyamagu/faiss-wheels/releases/download/v1.7.3/faiss_gpu-1.7.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install faiss-cpu==1.7.3
```

Next, we install PyTorch Geometric (CPU version). Note that the libraries need to be installed in the specified order:

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install pyg-lib -f https://data.pyg.org/whl/nightly/torch-2.4.0+cpu.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

Finally, we install the remaining libraries:

```bash
pip install -r requirements.txt
```

## Training recommender systems

After setting up the environment, we can train the recommender systems. The training scripts are located in the
`scripts` folder. It currently includes training scripts for the following models:
- `train_baseline.py`: Baseline model built on LightGCN architecture on a homogeneous graph of authors.
- `train_homogeneous.py`: A graph attention neural network (GAT) model on a homogeneous graph of authors including their
research interests through aggregated article embeddings.
- `train_heterogeneous.py`: 


### Running the FastAPI server for model inference

We run the FastAPI server for model inference. The API can be run using the following command:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```