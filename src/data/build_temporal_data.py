"""Build temporal edge data for TGN-style models."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch


def load_features(features_path: Path) -> pd.DataFrame:
    with open(features_path, "r", encoding="utf-8") as f:
        first = f.readline().strip()
    if first.startswith("txId,"):
        return pd.read_csv(features_path)
    features = pd.read_csv(features_path, header=None)
    n_cols = features.shape[1]
    feature_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(n_cols - 2)]
    features.columns = feature_cols
    return features


def load_edges(edges_path: Path) -> pd.DataFrame:
    return pd.read_csv(edges_path)


def build_id_mapping(raw_ids: np.ndarray) -> Dict[int, int]:
    return {int(raw_id): i for i, raw_id in enumerate(raw_ids)}


def build_temporal_data(
    data_dir: Path,
    *,
    features_path: Path | None = None,
    classes_path: Path | None = None,
) -> dict:
    if features_path is None:
        features_path = data_dir / "elliptic_txs_features.csv"
    edges_path = data_dir / "elliptic_txs_edgelist.csv"

    features = load_features(features_path)
    edges = load_edges(edges_path)
    if classes_path is None:
        classes_path = data_dir / "elliptic_txs_classes.csv"
    classes = pd.read_csv(classes_path)
    class_map = {1: 1, 2: 0, "1": 1, "2": 0}
    classes["label"] = classes["class"].map(class_map)

    raw_ids = features["txId"].to_numpy()
    id_mapping = build_id_mapping(raw_ids)

    x = torch.tensor(
        features.drop(columns=["txId", "time_step"]).to_numpy(),
        dtype=torch.float32,
    )
    time_step = torch.tensor(features["time_step"].to_numpy(), dtype=torch.long)

    src = edges["txId1"].map(id_mapping)
    dst = edges["txId2"].map(id_mapping)
    mask = src.notna() & dst.notna()
    src_idx = src[mask].astype(int).to_numpy()
    dst_idx = dst[mask].astype(int).to_numpy()

    # Use destination time as event time (fallback to max of src/dst)
    src_time = time_step[src_idx].numpy()
    dst_time = time_step[dst_idx].numpy()
    edge_time = np.maximum(src_time, dst_time)

    # Message features: source node features (simple choice)
    msg = x[src_idx]

    train_mask = (time_step <= 34)
    val_mask = ((time_step >= 35) & (time_step <= 41))
    test_mask = (time_step >= 42)

    payload = {
        "src": torch.tensor(src_idx, dtype=torch.long),
        "dst": torch.tensor(dst_idx, dtype=torch.long),
        "t": torch.tensor(edge_time, dtype=torch.long),
        "msg": msg,
        "x": x,
        "time_step": time_step,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "id_mapping": id_mapping,
        "y": torch.tensor(
            features["txId"].map(classes.set_index("txId")["label"]).fillna(-1).astype(int).to_numpy(),
            dtype=torch.long,
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal data for TGN")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output", default="data/processed/temporal_data.pt")
    parser.add_argument(
        "--features-path",
        default=None,
        help="Optional CSV with engineered features (must include txId and time_step)",
    )
    args = parser.parse_args()

    features_path = Path(args.features_path) if args.features_path else None
    payload = build_temporal_data(Path(args.data_dir), features_path=features_path)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved temporal data to {output_path}")


if __name__ == "__main__":
    main()
