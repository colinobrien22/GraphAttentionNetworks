# src/graph_attention_networks/model.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """
    Two-layer GAT for node classification (Cora).
    - Layer 1: multi-head attention, concatenated, ELU
    - Layer 2: multi-head attention, averaged (concat=False), logits
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 8,
        out_channels: int = 7,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        self.dropout_p = dropout

        # First attention layer: concat heads -> output dim = hidden * heads
        self.conv1 = GATConv(
            in_channels,
            hidden,
            heads=heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )

        # Second attention layer: no concat -> output dim = out_channels
        self.conv2 = GATConv(
            hidden * heads,
            out_channels,
            heads=1,  # one head on output (common GAT setup)
            concat=False,  # average, produce class logits
            dropout=dropout,
            add_self_loops=True,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Feature dropout
        x = self.dropout(x)
        # First GAT layer + nonlinearity
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        # Dropout between layers
        x = self.dropout(x)
        # Output layer -> logits (no softmax; use CrossEntropyLoss)
        x = self.conv2(x, edge_index)
        return x
