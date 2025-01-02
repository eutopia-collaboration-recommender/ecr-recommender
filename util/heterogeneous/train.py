import torch
from torch.optim import Optimizer
from torch_geometric.data import HeteroData
from torch_geometric.metrics.link_pred import (
    LinkPredMAP,
    LinkPredMRR,
    LinkPredNDCG,
    LinkPredPrecision,
    LinkPredRecall
)
from torch_geometric.utils import structured_negative_sampling

from util.heterogeneous.model import ModelEuCoHT
from util.torch_geometric import LinkPredHitRate


def train(model: ModelEuCoHT,
          data: HeteroData,
          optimizer: Optimizer,
          target_edge_type: tuple,
          target_node_type: str) -> float:
    model.train()
    # Fetch existing graph edges as positive samples
    pos_edge_index = data[target_edge_type].train_edge_index
    # Perform structured negative sampling, meaning that for every positive sample,
    # there will be another negative sample at the same index/position
    # - this structure is expected in BPR loss calculation
    neg_edge_index_i, neg_edge_index_j, neg_edge_index_k = structured_negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data[target_node_type].num_nodes)
    # Structure negative sampling returns both positive and negative edges, so
    # we need to concatenate appropriate nodes to get the negative edges
    neg_edge_index = torch.stack([neg_edge_index_i, neg_edge_index_k], dim=0)

    optimizer.zero_grad()

    # Concatenate edge label indices into a single edge label index
    edge_label_index = torch.cat([
        pos_edge_index,
        neg_edge_index,
    ], dim=1)

    pos_edge_rank, neg_edge_rank = model(x_dict=data.x_dict,
                                         edge_index_dict=data.train_edge_index_dict,
                                         edge_label_index=edge_label_index).chunk(2)

    # Calculate BPR loss
    loss = model.recommendation_loss(x_dict=data.x_dict,
                                     edge_index_dict=data.train_edge_index_dict,
                                     pos_edge_rank=pos_edge_rank,
                                     neg_edge_rank=neg_edge_rank,
                                     node_id=edge_label_index.unique())

    loss.backward()
    optimizer.step()

    total_loss = float(loss) * pos_edge_rank.numel()
    total_examples = pos_edge_rank.numel()

    # Cleanup
    del pos_edge_rank, neg_edge_rank
    torch.cuda.empty_cache()

    return total_loss / total_examples


@torch.no_grad()
def test(model: ModelEuCoHT,
         data: HeteroData,
         target_edge_type: tuple):
    model.eval()

    pos_edge_index = data[target_edge_type].test_pos_edge_index
    neg_edge_index = data[target_edge_type].test_neg_edge_index

    # Concatenate edge label indices into a single edge label index
    edge_label_index = torch.cat([
        pos_edge_index,
        neg_edge_index,
    ], dim=1)

    # We encode the training edge index to get the embeddings based on the training graph
    # structure and then use those embeddings to predict unseen edges
    pos_edge_rank, neg_edge_rank = model(x_dict=data.x_dict,
                                         edge_index_dict=data.train_edge_index_dict,
                                         edge_label_index=edge_label_index).chunk(2)

    # Calculate BPR loss
    loss = model.recommendation_loss(x_dict=data.x_dict,
                                     edge_index_dict=data.train_edge_index_dict,
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
def evaluate(model: ModelEuCoHT,
             data: HeteroData,
             target_edge_type: tuple,
             target_node_type: str,
             num_recommendations: int,
             device: str = 'cpu'):
    model.eval()
    z_dict = model.encoder(data.x_dict, data.train_edge_index_dict)
    embs = z_dict[target_node_type].to(device)

    result = {
        'precision@k': LinkPredPrecision(k=num_recommendations).to(device),
        'recall@k': LinkPredRecall(k=num_recommendations).to(device),
        'map@k': LinkPredMAP(k=num_recommendations).to(device),
        'mrr@k': LinkPredMRR(k=num_recommendations).to(device),
        'ndcg@k': LinkPredNDCG(k=num_recommendations).to(device),
        'hit_rate@k': LinkPredHitRate(k=num_recommendations).to(device)
    }

    # Calculate distance between embeddings
    logits = embs @ embs.T

    # Exclude training edges
    logits[data[target_edge_type].train_edge_index[0], data[target_edge_type].train_edge_index[1]] = float('-inf')

    # Gather ground truth data
    ground_truth = data[target_edge_type].test_pos_edge_index

    # Get top-k recommendations for each node
    top_k_index = torch.topk(logits, k=num_recommendations, dim=1).indices

    # Update performance metrics
    for metric in result.keys():
        result[metric].update(
            pred_index_mat=top_k_index,
            edge_label_index=ground_truth)

    # Cleanup
    del embs, logits, ground_truth, top_k_index
    torch.cuda.empty_cache()

    return result
