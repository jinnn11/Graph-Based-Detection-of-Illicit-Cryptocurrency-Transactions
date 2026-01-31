"""Feature engineering for tabular and GNN baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_features(features_path: Path) -> pd.DataFrame:
    df = pd.read_csv(features_path, header=None)
    n_cols = df.shape[1]
    cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(n_cols - 2)]
    df.columns = cols
    return df


def load_edges(edges_path: Path) -> pd.DataFrame:
    return pd.read_csv(edges_path)


def add_degree_features(features: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    out_deg = edges["txId1"].value_counts()
    in_deg = edges["txId2"].value_counts()

    features = features.copy()
    features["out_degree"] = features["txId"].map(out_deg).fillna(0).astype(int)
    features["in_degree"] = features["txId"].map(in_deg).fillna(0).astype(int)
    features["total_degree"] = features["in_degree"] + features["out_degree"]

    features["log1p_out_degree"] = np.log1p(features["out_degree"].to_numpy())
    features["log1p_in_degree"] = np.log1p(features["in_degree"].to_numpy())
    features["log1p_total_degree"] = np.log1p(features["total_degree"].to_numpy())
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate engineered feature table")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument(
        "--output",
        default="data/processed/tabular_features.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--add-degree",
        action="store_true",
        help="Add degree-based features",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    features = load_features(data_dir / "elliptic_txs_features.csv")

    if args.add_degree:
        edges = load_edges(data_dir / "elliptic_txs_edgelist.csv")
        features = add_degree_features(features, edges)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}")


if __name__ == "__main__":
    main()
