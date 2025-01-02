import torch

from torch.optim import Optimizer
from torch_geometric.data import Data
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.metrics.link_pred import (
    LinkPredMAP,
    LinkPredMRR,
    LinkPredNDCG,
    LinkPredPrecision,
    LinkPredRecall
)

from util.homogeneous.model import ModelEuCoHM
from util.torch_geometric import LinkPredHitRate


def train(model: ModelEuCoHM,
          data: Data,
          optimizer: Optimizer) -> float:
    # Set the model to training mode
    model.train()

    # Get the positive edge index
    pos_edge_index = data.train_pos_edge_index
    # Negative sampling
    neg_edge_index_i, neg_edge_index_j, neg_edge_index_k = structured_negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes)
    neg_edge_index = torch.stack([neg_edge_index_i, neg_edge_index_k], dim=0)

    # Reset the optimizer
    optimizer.zero_grad()

    # Concatenate edge label indices into a single edge label index
    edge_label_index = torch.cat([
        pos_edge_index,
        neg_edge_index,
    ], dim=1)

    # Get the positive and negative edge ranks
    pos_edge_rank, neg_edge_rank = model(x=data.x,
                                         edge_index=data.train_pos_edge_index,
                                         edge_label_index=edge_label_index).chunk(2)

    # Calculate BPR loss
    loss = model.recommendation_loss(x=data.x,
                                     edge_index=data.train_pos_edge_index,
                                     pos_edge_rank=pos_edge_rank,
                                     neg_edge_rank=neg_edge_rank,
                                     node_id=edge_label_index.unique())

    loss.backward()
    optimizer.step()

    # Calculate the total loss
    total_loss = float(loss) * pos_edge_rank.numel()
    total_examples = pos_edge_rank.numel()

    # Cleanup
    del pos_edge_rank, neg_edge_rank
    torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def test(model: ModelEuCoHM,
         data: Data) -> float:
    model.eval()

    pos_edge_index = data.test_pos_edge_index
    neg_edge_index = data.test_neg_edge_index

    edge_label_index = torch.cat([
        pos_edge_index,
        neg_edge_index,
    ], dim=1)

    # We encode the training edge index to get the embeddings based on the training graph
    # structure and then use those embeddings to predict unseen edges
    pos_edge_rank, neg_edge_rank = model(x=data.x,
                                         edge_index=data.train_pos_edge_index,
                                         edge_label_index=edge_label_index).chunk(2)

    # Calculate BPR loss
    loss = model.recommendation_loss(x=data.x,
                                     edge_index=data.train_pos_edge_index,
                                     pos_edge_rank=pos_edge_rank,
                                     neg_edge_rank=neg_edge_rank,
                                     node_id=edge_label_index.unique())

    total_loss = float(loss) * pos_edge_rank.numel()
    total_examples = pos_edge_rank.numel()

    # Cleanup
    del pos_edge_rank, neg_edge_rank
    torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def evaluate(model: ModelEuCoHM,
             data: Data,
             device: str = 'cpu',
             k: int = 20) -> dict:
    model.eval()
    embs = model.get_embedding(x=data.x, edge_index=data.train_pos_edge_index).to(device)

    result = {
        'precision@k': LinkPredPrecision(k=k).to(device),
        'recall@k': LinkPredRecall(k=k).to(device),
        'map@k': LinkPredMAP(k=k).to(device),
        'mrr@k': LinkPredMRR(k=k).to(device),
        'ndcg@k': LinkPredNDCG(k=k).to(device),
        'hit_rate@k': LinkPredHitRate(k=k).to(device)
    }

    # Calculate distance between embeddings
    logits = embs @ embs.T

    # Exclude training edges
    logits[data.train_pos_edge_index[0], data.train_pos_edge_index[1]] = float('-inf')

    # Gather ground truth data
    ground_truth = data.test_pos_edge_index

    # Get top-k recommendations for each node
    top_k_index = torch.topk(logits, k=k, dim=1).indices

    # Update performance metrics
    for metric in result.keys():
        result[metric].update(
            pred_index_mat=top_k_index,
            edge_label_index=ground_truth)

    # Cleanup
    del embs, logits, ground_truth, top_k_index
    torch.cuda.empty_cache()

    return result
