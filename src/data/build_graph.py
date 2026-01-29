"""Build PyTorch Geometric-ready tensors from the Elliptic dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    time_step: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    id_mapping: Dict[int, int]
    x_mean: Optional[torch.Tensor] = None
    x_std: Optional[torch.Tensor] = None
    add_reverse_edges: bool = False
    normalized: bool = False


def load_features(features_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path, header=None)
    n_cols = features.shape[1]
    feature_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(n_cols - 2)]
    features.columns = feature_cols
    return features


def load_edges(edges_path: Path) -> pd.DataFrame:
    return pd.read_csv(edges_path)


def build_id_mapping(raw_ids: np.ndarray) -> Dict[int, int]:
    return {int(raw_id): i for i, raw_id in enumerate(raw_ids)}


def map_edges(edges: pd.DataFrame, id_mapping: Dict[int, int]) -> torch.Tensor:
    src = edges["txId1"].map(id_mapping)
    dst = edges["txId2"].map(id_mapping)
    mask = src.notna() & dst.notna()

    src_idx = src[mask].astype(int).to_numpy()
    dst_idx = dst[mask].astype(int).to_numpy()

    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
    return edge_index


def add_reverse_edges(edge_index: torch.Tensor) -> torch.Tensor:
    rev = edge_index.flip(0)
    return torch.cat([edge_index, rev], dim=1)


def normalize_features(
    x: torch.Tensor, train_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = x[train_mask]
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    x_norm = (x - mean) / std
    return x_norm, mean, std


def build_splits(time_step: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = (time_step <= 34).to_numpy()
    val_mask = ((time_step >= 35) & (time_step <= 41)).to_numpy()
    test_mask = (time_step >= 42).to_numpy()

    return (
        torch.tensor(train_mask, dtype=torch.bool),
        torch.tensor(val_mask, dtype=torch.bool),
        torch.tensor(test_mask, dtype=torch.bool),
    )


def build_graph(
    data_dir: Path,
    *,
    add_reverse: bool = False,
    normalize: bool = False,
) -> GraphData:
    features_path = data_dir / "elliptic_txs_features.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    features = load_features(features_path)
    edges = load_edges(edges_path)

    raw_ids = features["txId"].to_numpy()
    id_mapping = build_id_mapping(raw_ids)

    x = torch.tensor(
        features.drop(columns=["txId", "time_step"]).to_numpy(),
        dtype=torch.float32,
    )
    time_step = torch.tensor(features["time_step"].to_numpy(), dtype=torch.long)
    edge_index = map_edges(edges, id_mapping)

    train_mask, val_mask, test_mask = build_splits(features["time_step"])

    x_mean = None
    x_std = None
    if normalize:
        x, x_mean, x_std = normalize_features(x, train_mask)

    if add_reverse:
        edge_index = add_reverse_edges(edge_index)

    return GraphData(
        x=x,
        edge_index=edge_index,
        time_step=time_step,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        id_mapping=id_mapping,
        x_mean=x_mean,
        x_std=x_std,
        add_reverse_edges=add_reverse,
        normalized=normalize,
    )


def save_graph(graph: GraphData, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": graph.x,
        "edge_index": graph.edge_index,
        "time_step": graph.time_step,
        "train_mask": graph.train_mask,
        "val_mask": graph.val_mask,
        "test_mask": graph.test_mask,
        "id_mapping": graph.id_mapping,
        "x_mean": graph.x_mean,
        "x_std": graph.x_std,
        "add_reverse_edges": graph.add_reverse_edges,
        "normalized": graph.normalized,
    }
    torch.save(payload, output_path)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build processed graph tensors")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--output", default="data/processed/graph.pt", help="Output .pt path")
    parser.add_argument(
        "--add-reverse-edges",
        action="store_true",
        help="Add reverse edges to make message passing bidirectional",
    )
    parser.add_argument(
        "--normalize-features",
        action="store_true",
        help="Standardize features using train split statistics",
    )
    args = parser.parse_args()

    graph = build_graph(
        Path(args.data_dir),
        add_reverse=args.add_reverse_edges,
        normalize=args.normalize_features,
    )
    save_graph(graph, Path(args.output))
    print("Saved:", args.output)
    print("x:", graph.x.shape)
    print("edge_index:", graph.edge_index.shape)
    print(
        "train/val/test:",
        graph.train_mask.sum().item(),
        graph.val_mask.sum().item(),
        graph.test_mask.sum().item(),
    )


if __name__ == "__main__":
    main()
