import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool


class PGroupGNN(torch.nn.Module):
    def __init__(self, node_dim=5, hidden_dim=128, ph_dim=100, out_dim=5,
                 pooling="mean"):
        """
        node_dim: number of raw per-element node features:
            (element order, centralizer size, conjugacy class size,
            centrality indicator, power depth).
        out_dim: number of regression targets (nilpotency class, derived
            length, centre size, commutator size, number of conjugacy
            classes). Must match the length of the target vector y built in
            dataset.py.
        pooling: "mean" (default), "sum", or "mean+sum". "mean+sum"
            concatenates the mean-pooled and sum-pooled graph embeddings.
            The Weisfeiler-Lehman expressiveness result for GIN (Xu et al.)
            assumes sum aggregation and a sum readout, so it does not cover
            the default mean readout.
        """
        super().__init__()
        valid_pooling = {"mean", "sum", "mean+sum"}
        if pooling not in valid_pooling:
            raise ValueError(
                f"pooling must be one of {sorted(valid_pooling)}, got {pooling!r}")
        self.pooling = pooling

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(node_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        self.mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GINConv(self.mlp1)
        self.conv2 = GINConv(self.mlp2)
        self.conv3 = GINConv(self.mlp3)

        pooled_mult = 2 if pooling == "mean+sum" else 1
        self.fc1 = torch.nn.Linear(hidden_dim * pooled_mult + ph_dim, 128)
        self.fc2 = torch.nn.Linear(128, out_dim)

        self.ph_dim = ph_dim

    def forward(self, x, edge_index, batch, ph_features):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        if self.pooling == "sum":
            x = global_add_pool(x, batch)
        elif self.pooling == "mean+sum":
            x = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
        else:
            x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]

        if self.ph_dim > 0:
            ph_features = ph_features.view(-1, self.ph_dim)
            x = torch.cat([x, ph_features], dim=1)

        x = F.relu(self.fc1(x))
        return self.fc2(x)
