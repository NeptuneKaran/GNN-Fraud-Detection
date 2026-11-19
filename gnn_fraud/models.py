import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


CONVOLUTIONS = {"GraphSAGE": SAGEConv, "GCN": GCNConv, "GAT": GATConv}


class TransactionGNN(nn.Module):
    def __init__(self, model_name: str, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        convolution = CONVOLUTIONS[model_name]
        self.input = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            convolution(hidden_channels, hidden_channels) for _ in range(2)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in self.convs)
        self.output = nn.Linear(hidden_channels, 2)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.input(x))
        for convolution, norm in zip(self.convs, self.norms):
            update = self.dropout(F.gelu(convolution(x, edge_index)))
            x = norm(x + update)
        return self.output(x)
