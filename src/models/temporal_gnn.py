"""Temporal GNN models with time encoding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "torch_geometric is required for temporal GNNs. "
        "Install with `pip install torch-geometric`."
    ) from exc


@dataclass
class TemporalGATConfig:
    in_channels: int
    time_dim: int
    hidden_channels: int
    out_channels: int
    num_layers: int = 2
    dropout: float = 0.5
    heads: int = 4
    max_time: int = 50


class TemporalGAT(torch.nn.Module):
    def __init__(self, config: TemporalGATConfig) -> None:
        super().__init__()
        self.dropout = config.dropout
        self.time_emb = torch.nn.Embedding(config.max_time + 1, config.time_dim)

        layers = []
        in_dim = config.in_channels + config.time_dim
        layers.append(GATv2Conv(in_dim, config.hidden_channels, heads=config.heads, dropout=config.dropout))
        for _ in range(config.num_layers - 2):
            layers.append(
                GATv2Conv(
                    config.hidden_channels * config.heads,
                    config.hidden_channels,
                    heads=config.heads,
                    dropout=config.dropout,
                )
            )
        layers.append(
            GATv2Conv(
                config.hidden_channels * config.heads,
                config.out_channels,
                heads=1,
                concat=False,
                dropout=config.dropout,
            )
        )
        self.convs = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, time_step: torch.Tensor) -> torch.Tensor:
        t = self.time_emb(time_step)
        h = torch.cat([x, t], dim=1)
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.convs[-1](h, edge_index)
        return h


def build_temporal_gat(
    in_channels: int,
    time_dim: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    heads: int = 4,
    max_time: int = 50,
) -> TemporalGAT:
    config = TemporalGATConfig(
        in_channels=in_channels,
        time_dim=time_dim,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
        heads=heads,
        max_time=max_time,
    )
    return TemporalGAT(config)
