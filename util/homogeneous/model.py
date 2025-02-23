import torch

from typing import Optional

from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.models.lightgcn import BPRLoss


class ModelEuCoHM(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 num_recommendations: int,
                 author_node_id_map: dict,
                 author_id_map: dict,
                 device: str = 'cpu'):
        super().__init__()
        # Set the parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_recommendations = num_recommendations
        self.device = device

        # Set mapper to contiguous ids
        self.author_node_id_map: dict = author_node_id_map
        self.author_id_map: dict = author_id_map

        # Initialize the convolutional layers
        self.conv_layers = ModuleList([
            GATv2Conv(
                in_channels=self.input_channels if ix == 0 else self.hidden_channels,
                out_channels=self.hidden_channels,
                add_self_loops=True,
                negative_slope=0.2
            )
            for ix in range(self.num_layers)
        ])

        # Initialize batch norm layers
        self.bn_layers = ModuleList([
            torch.nn.BatchNorm1d(self.hidden_channels)
            for ix in range(self.num_layers)
        ])

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

        out = x * self.alpha[0]

        # Activate all other layers
        for i in range(1, self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.bn_layers[i](x)
            out = out + x * self.alpha[i + 1]

        return out

    def recommend(self,
                  x: Tensor,
                  edge_index: Adj,
                  author_id: str) -> list:

        # Get all embeddings
        out_src = out_dst = self.get_embedding(x, edge_index)

        # Get the author id
        author_id = self.author_node_id_map[author_id]
        # Get the author embedding
        out_src = out_src[author_id]

        # Calculate the dot product
        pred = out_src @ out_dst.t()
        # Get the top k recommendations
        top_index = pred.topk(self.num_recommendations, dim=-1, sorted=True).indices

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
                            edge_weight: Optional[Tensor] = None,
                            lambda_reg: float = 1e-4) -> Tensor:
        loss_fn = BPRLoss(lambda_reg)
        # Get the embedding
        emb = self.get_embedding(x=x, edge_index=edge_index)
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)
