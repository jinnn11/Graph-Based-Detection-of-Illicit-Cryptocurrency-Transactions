"""Simple GNN models for node classification on the Elliptic graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv, GATv2Conv, GCNConv, SAGEConv
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "torch_geometric is required for the GNN baseline. "
        "Install with `pip install torch-geometric` and the correct PyG wheels."
    ) from exc


ModelType = Literal["gcn", "sage", "gat", "gatv2"]


@dataclass
class GNNConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int = 2
    dropout: float = 0.5
    model_type: ModelType = "sage"
    heads: int = 4


class SimpleGNN(torch.nn.Module):
    def __init__(self, config: GNNConfig) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.model_type = config.model_type

        layers = []
        if config.model_type == "gcn":
            layers.append(GCNConv(config.in_channels, config.hidden_channels))
            for _ in range(config.num_layers - 2):
                layers.append(GCNConv(config.hidden_channels, config.hidden_channels))
            layers.append(GCNConv(config.hidden_channels, config.out_channels))
        elif config.model_type == "gat":
            heads = config.heads
            layers.append(GATConv(config.in_channels, config.hidden_channels, heads=heads, dropout=config.dropout))
            for _ in range(config.num_layers - 2):
                layers.append(
                    GATConv(
                        config.hidden_channels * heads,
                        config.hidden_channels,
                        heads=heads,
                        dropout=config.dropout,
                    )
                )
            layers.append(
                GATConv(
                    config.hidden_channels * heads,
                    config.out_channels,
                    heads=1,
                    concat=False,
                    dropout=config.dropout,
                )
            )
        elif config.model_type == "gatv2":
            heads = config.heads
            layers.append(GATv2Conv(config.in_channels, config.hidden_channels, heads=heads, dropout=config.dropout))
            for _ in range(config.num_layers - 2):
                layers.append(
                    GATv2Conv(
                        config.hidden_channels * heads,
                        config.hidden_channels,
                        heads=heads,
                        dropout=config.dropout,
                    )
                )
            layers.append(
                GATv2Conv(
                    config.hidden_channels * heads,
                    config.out_channels,
                    heads=1,
                    concat=False,
                    dropout=config.dropout,
                )
            )
        else:
            layers.append(SAGEConv(config.in_channels, config.hidden_channels))
            for _ in range(config.num_layers - 2):
                layers.append(SAGEConv(config.hidden_channels, config.hidden_channels))
            layers.append(SAGEConv(config.hidden_channels, config.out_channels))

        self.convs = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def build_gnn(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    model_type: ModelType = "sage",
    heads: int = 4,
) -> SimpleGNN:
    config = GNNConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type,
        heads=heads,
    )
    return SimpleGNN(config)
