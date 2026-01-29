"""Hyperparameter sweep runner for GNN baselines."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.training.train_gnn import TrainConfig, run_experiment


def build_grid(model: str) -> List[Dict[str, Any]]:
    if model in {"gat", "gatv2"}:
        grid = {
            "hidden_dim": [64, 128],
            "num_layers": [2, 3],
            "dropout": [0.3, 0.5],
            "lr": [0.005, 0.003],
            "weight_decay": [5e-4, 1e-3],
            "heads": [2, 4, 8],
            "loss": ["cross_entropy", "focal"],
            "focal_gamma": [1.0, 2.0],
        }
    else:
        grid = {
            "hidden_dim": [64, 128, 256],
            "num_layers": [2, 3],
            "dropout": [0.3, 0.5],
            "lr": [0.01, 0.005],
            "weight_decay": [5e-4, 1e-3],
            "heads": [4],
            "loss": ["cross_entropy", "focal"],
            "focal_gamma": [1.0, 2.0],
        }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def sample_trials(grid: List[Dict[str, Any]], max_trials: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    if max_trials >= len(grid):
        return grid
    indices = rng.choice(len(grid), size=max_trials, replace=False)
    return [grid[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="GNN hyperparameter sweep")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--graph-path", default="data/processed/graph.pt")
    parser.add_argument("--models", choices=["sage", "gat", "gatv2", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--output-dir", default="experiments/gnn_sweep")
    parser.add_argument("--metric", default="pr_auc")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    models = ["sage", "gat", "gatv2"] if args.models == "all" else [args.models]

    leaderboard = []
    trial_id = 0
    for model in models:
        grid = build_grid(model)
        trials = sample_trials(grid, args.max_trials, rng)
        print(f"Running {len(trials)} trials for model={model}")

        for params in trials:
            trial_id += 1
            output_path = output_dir / f"{model}_trial_{trial_id}.json"
            model_output = model_dir / f"{model}_trial_{trial_id}.pt"

            config = TrainConfig(
                data_dir=args.data_dir,
                graph_path=args.graph_path,
                epochs=args.epochs,
                hidden_dim=params["hidden_dim"],
                num_layers=params["num_layers"],
                dropout=params["dropout"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                model=model,
                heads=params.get("heads", 4),
                seed=args.seed,
                output=str(output_path),
                model_output=str(model_output),
                patience=args.patience,
                loss=params.get("loss", "cross_entropy"),
                focal_gamma=params.get("focal_gamma", 2.0),
                add_self_loops=True,
            )

            print(f"\nTrial {trial_id} | model={model} | params={params}")
            metrics = run_experiment(config)
            leaderboard.append({
                "output": str(output_path),
                "model": model,
                "params": params,
                "metrics": metrics,
            })

    metric = args.metric
    leaderboard = [row for row in leaderboard if metric in row["metrics"]]
    leaderboard.sort(key=lambda x: x["metrics"][metric], reverse=True)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(leaderboard, indent=2))

    print("\nLeaderboard (top 10):")
    for rank, row in enumerate(leaderboard[:10], start=1):
        score = row["metrics"][metric]
        print(f"{rank:02d}. {row['model']} {metric}={score:.4f} ({row['output']})")


if __name__ == "__main__":
    main()
