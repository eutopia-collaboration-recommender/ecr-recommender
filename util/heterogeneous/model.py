from typing import Optional

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.nn import Linear, to_hetero

from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.typing import OptTensor
from torch_geometric.nn.models.lightgcn import BPRLoss


class Encoder(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 include_linear_layers: bool,
                 include_activation_layers: bool
                 ):
        super().__init__()

        self.num_layers = num_layers
        self.include_linear_layers = include_linear_layers
        self.include_activation_layers = include_activation_layers
        self.alpha = 1. / (self.num_layers + 1)  # not used ATM

        # # Initialize the convolutional layers
        self.conv_layers = ModuleList([
            GATv2Conv(
                in_channels=(-1, -1),
                out_channels=hidden_channels,
                add_self_loops=False,
                negative_slope=0.2
            )
            for ix in range(self.num_layers)
        ])

        # Linear layers
        self.lin_layers = ModuleList([
            Linear(-1, hidden_channels) for i in range(self.num_layers)

        ])

        # Initialize batch norm layers
        self.bn_layers = ModuleList([
            torch.nn.BatchNorm1d(hidden_channels) for i in range(self.num_layers)
        ])

    def forward(self, x, edge_index):
        # Activate layers
        for i in range(self.num_layers):
            x = self.conv_layers[i](x, edge_index)
            # Linear layer (skip connection instead of self loops)
            if self.include_linear_layers:
                x += self.lin_layers[i](x)
            x = self.bn_layers[i](x)
            # Activation layer
            if self.include_activation_layers and i < self.num_layers - 1:
                x = x.relu()

        return x


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z, edge_label_index):
        out_src = z[edge_label_index[0]]
        out_dst = z[edge_label_index[1]]

        # Calculate the dot product
        return (out_src * out_dst).sum(dim=-1)


class ModelEuCoHT(torch.nn.Module):
    def __init__(self,
                 hidden_channels: int,
                 num_layers: int,
                 include_linear_layers: bool,
                 include_activation_layers: bool,
                 num_recommendations: int,
                 author_node_id_map: dict,
                 author_id_map: dict):
        super().__init__()

        self.hidden_channels: int = hidden_channels
        self.num_recommendations: int = num_recommendations
        self.num_layers: int = num_layers
        self.include_linear_layers: int = include_linear_layers
        self.include_activation_layers: int = include_activation_layers
        self.author_node_id_map: dict = author_node_id_map
        self.author_id_map: dict = author_id_map
        self.target_node_type: str

        self.encoder = Encoder(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            include_linear_layers=include_linear_layers,
            include_activation_layers=include_activation_layers
        )
        self.encoder = to_hetero(self.encoder, (['article', 'author'],
                                                [('author', 'publishes', 'article'),
                                                 ('author', 'co_authors', 'author'),
                                                 ('article', 'rev_publishes', 'author')]), aggr='sum')
        self.decoder = Decoder()

    def forward(self,
                x_dict,
                edge_index_dict,
                edge_label_index: OptTensor = None):
        # Encode authors
        z_dict = self.encoder(x_dict, edge_index_dict)
        z = z_dict['author']
        # Decode edges
        return self.decoder(z, edge_label_index)

    def recommend(self,
                  x_dict,
                  edge_index_dict,
                  author_sid: str,
                  k: int = 10):
        # Get all embeddings
        out_src = out_dst = self.encoder(x_dict, edge_index_dict)

        # Get the author id
        author_id = self.author_node_id_map[author_sid]
        # Get the author embedding
        out_src = out_src[author_id]

        # Calculate the dot product
        pred = out_src @ out_dst.t()
        # Get the top k recommendations
        top_index = pred.topk(k, dim=-1, sorted=True).indices

        # Decode top k recommendations to author SIDs
        top_author_sids = [self.author_id_map[int(i)] for i in top_index]

        return top_author_sids

    def recommendation_loss(self,
                            x_dict,
                            edge_index_dict,
                            pos_edge_rank,
                            neg_edge_rank,
                            node_id: Optional[Tensor] = None,
                            lambda_reg: float = 1e-4):
        loss_fn = BPRLoss(lambda_reg)
        # Get the embedding
        z_dict = self.encoder(x_dict, edge_index_dict)
        emb = z_dict['author']
        # Get the loss
        emb = emb if node_id is None else emb[node_id]
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)
