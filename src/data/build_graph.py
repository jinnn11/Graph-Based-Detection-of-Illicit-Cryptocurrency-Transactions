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


def build_splits(time_step: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    train_mask = (time_step <= 34).to_numpy()
    val_mask = ((time_step >= 35) & (time_step <= 41)).to_numpy()
    test_mask = (time_step >= 42).to_numpy()

    return (
        torch.tensor(train_mask, dtype=torch.bool),
        torch.tensor(val_mask, dtype=torch.bool),
        torch.tensor(test_mask, dtype=torch.bool),
    )


def build_graph(data_dir: Path) -> GraphData:
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

    return GraphData(
        x=x,
        edge_index=edge_index,
        time_step=time_step,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        id_mapping=id_mapping,
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
    }
    torch.save(payload, output_path)


def main(data_dir: Optional[str] = None, output: Optional[str] = None) -> None:
    if data_dir is None:
        data_dir = "data/raw"
    if output is None:
        output = "data/processed/graph.pt"
    graph = build_graph(Path(data_dir))
    save_graph(graph, Path(output))
    print("Saved:", output)
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
