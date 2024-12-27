import psycopg2.extensions
import torch

from typing import Optional

from torch import Tensor
from torch.nn import ModuleList
from torch.optim import Optimizer
from torch_geometric.data import Data

from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.models.lightgcn import BPRLoss
from torch_geometric.utils import structured_negative_sampling
from torch_geometric.metrics.link_pred import (
    LinkPredMAP,
    LinkPredMRR,
    LinkPredNDCG,
    LinkPredPrecision,
    LinkPredRecall
)

from util.torch_geometric import LinkPredHitRate


class ModelEuCoHM(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 k: int,
                 author_node_id_map: dict,
                 author_id_map: dict):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.k = k

        # Set mapper to contiguous ids
        self.author_node_id_map: dict = author_node_id_map
        self.author_id_map: dict = author_id_map

        # Initialize the convolutional layers
        self.conv_layers = ModuleList([
            GATv2Conv(in_channels=input_channels, out_channels=hidden_channels, add_self_loops=True,
                      negative_slope=0.2),
            GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, add_self_loops=True,
                      negative_slope=0.2),
            GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, add_self_loops=True,
                      negative_slope=0.2),
            GATv2Conv(in_channels=hidden_channels, out_channels=hidden_channels, add_self_loops=True,
                      negative_slope=0.2),
        ])

        # Initialize batch norm layers
        self.bn_layers = ModuleList([
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
            torch.nn.BatchNorm1d(hidden_channels),
        ])

        # Calculate the number of layers
        self.num_layers = len(self.conv_layers)
        # Initialize the alpha
        alpha = 1. / (self.num_layers + 1)
        alpha = torch.tensor([alpha] * (self.num_layers + 1))
        self.register_buffer('alpha', alpha)

        # Initialize the parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize the convolutional layers
        for conv in self.conv_layers:
            conv.reset_parameters()

    def get_embedding(self,
                      x: Tensor,
                      edge_index: Adj) -> Tensor:
        # Activate first layer, since it is not of the same size as the target embedding
        x = self.conv_layers[0](x, edge_index)
        x = self.bn_layers[0](x)
        # x = F.relu(x)
        out = x * self.alpha[0]

        # Activate all other layers
        for i in range(1, self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            # x = F.relu(x)
            out = out + x * self.alpha[i + 1]

        return out

    def recommend(self,
                  x: Tensor,
                  edge_index: Adj,
                  author_id: str,
                  k: int = 10) -> list:

        # Get all embeddings
        out_src = out_dst = self.get_embedding(x, edge_index)

        # Get the author id
        author_id = self.author_node_id_map[author_id]
        # Get the author embedding
        out_src = out_src[author_id]

        # Calculate the dot product
        pred = out_src @ out_dst.t()
        # Get the top k recommendations
        top_index = pred.topk(k, dim=-1, sorted=True).indices

        # Decode top k recommendations to author SIDs
        top_author_ids = [self.author_id_map[int(i)] for i in top_index]

        return top_author_ids

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_label_index: OptTensor = None) -> Tensor:
        # Get the embedding
        out = self.get_embedding(x, edge_index)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        # Calculate the dot product
        return (out_src * out_dst).sum(dim=-1)

    def recommendation_loss(self,
                            x: Tensor,
                            edge_index: Adj,
                            pos_edge_rank: Tensor,
                            neg_edge_rank: Tensor,
                            node_id: Optional[Tensor] = None,
                            lambda_reg: float = 1e-4) -> Tensor:
        loss_fn = BPRLoss(lambda_reg)
        # Get the embedding
        emb = self.get_embedding(x=x, edge_index=edge_index)
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(hidden_layers={self.hidden_channels}, num_layers={self.num_layers})'


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
