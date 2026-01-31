"""Graph Transformer baseline using TransformerConv layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "torch_geometric is required for the Graph Transformer baseline."
    ) from exc


@dataclass
class GraphTransformerConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.5


class GraphTransformer(torch.nn.Module):
    def __init__(self, config: GraphTransformerConfig) -> None:
        super().__init__()
        self.dropout = config.dropout

        layers = []
        layers.append(
            TransformerConv(
                config.in_channels,
                config.hidden_channels,
                heads=config.heads,
                dropout=config.dropout,
                concat=True,
            )
        )
        for _ in range(config.num_layers - 2):
            layers.append(
                TransformerConv(
                    config.hidden_channels * config.heads,
                    config.hidden_channels,
                    heads=config.heads,
                    dropout=config.dropout,
                    concat=True,
                )
            )
        layers.append(
            TransformerConv(
                config.hidden_channels * config.heads,
                config.out_channels,
                heads=1,
                dropout=config.dropout,
                concat=False,
            )
        )
        self.convs = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


def build_graph_transformer(
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 2,
    heads: int = 4,
    dropout: float = 0.5,
) -> GraphTransformer:
    config = GraphTransformerConfig(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        heads=heads,
        dropout=dropout,
    )
    return GraphTransformer(config)
