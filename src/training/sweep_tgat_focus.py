"""Focused sweep around a previous TGATv2 trial."""
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

from src.training.train_tgat import TrainConfig, get_device, prepare_data, run_experiment


def build_focus_grid(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    hidden_dim = base.get("hidden_dim", 128)
    heads = base.get("heads", 4)
    time_dim = base.get("time_dim", 16)
    lr = base.get("lr", 0.003)
    dropout = base.get("dropout", 0.5)
    num_layers = base.get("num_layers", 2)
    weight_decay = base.get("weight_decay", 1e-3)
    loss = base.get("loss", "cross_entropy")
    focal_gamma = base.get("focal_gamma", 2.0)

    grid = {
        "hidden_dim": sorted({hidden_dim, max(32, hidden_dim // 2), hidden_dim * 2}),
        "heads": sorted({heads, max(1, heads // 2), heads * 2}),
        "time_dim": sorted({time_dim, max(4, time_dim // 2), time_dim * 2}),
        "lr": sorted({lr, lr * 0.5, lr * 1.5}),
        "dropout": sorted({dropout, max(0.1, dropout - 0.2), min(0.7, dropout + 0.2)}),
        "num_layers": sorted({num_layers, max(2, num_layers + 1)}),
        "weight_decay": sorted({weight_decay, weight_decay * 0.5, weight_decay * 2}),
        "loss": [loss],
        "focal_gamma": [focal_gamma],
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
    parser = argparse.ArgumentParser(description="Focused TGATv2 sweep")
    parser.add_argument("--base", required=True, help="Path to a TGAT result JSON")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--graph-path", default="data/processed/graph.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-trials", type=int, default=12)
    parser.add_argument("--output-dir", default="experiments/tgat_focus")
    parser.add_argument("--metric", default="pr_auc")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device selection (auto prefers cuda then mps)",
    )
    args = parser.parse_args()

    base_payload = json.loads(Path(args.base).read_text())
    base_config = base_payload.get("config", {})

    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    grid = build_focus_grid(base_config)
    trials = sample_trials(grid, args.max_trials, rng)
    print(f"Running {len(trials)} focused trials")

    device_pref = None if args.device == "auto" else args.device
    cached_data = prepare_data(
        TrainConfig(
            data_dir=args.data_dir,
            graph_path=args.graph_path,
            epochs=args.epochs,
            hidden_dim=base_config.get("hidden_dim", 128),
            num_layers=base_config.get("num_layers", 2),
            dropout=base_config.get("dropout", 0.5),
            lr=base_config.get("lr", 0.003),
            weight_decay=base_config.get("weight_decay", 1e-3),
            heads=base_config.get("heads", 4),
            time_dim=base_config.get("time_dim", 16),
            seed=args.seed,
            output=str(output_dir / "_warmup.json"),
            model_output=str(model_dir / "_warmup.pt"),
            patience=args.patience,
            loss=base_config.get("loss", "cross_entropy"),
            focal_gamma=base_config.get("focal_gamma", 2.0),
            add_self_loops=True,
            device=device_pref,
        ),
        get_device(device_pref),
    )

    leaderboard = []
    for idx, params in enumerate(trials, start=1):
        output_path = output_dir / f"tgat_focus_{idx}.json"
        model_output = model_dir / f"tgat_focus_{idx}.pt"

        config = TrainConfig(
            data_dir=args.data_dir,
            graph_path=args.graph_path,
            epochs=args.epochs,
            hidden_dim=params["hidden_dim"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            heads=params["heads"],
            time_dim=params["time_dim"],
            seed=args.seed,
            output=str(output_path),
            model_output=str(model_output),
            patience=args.patience,
            loss=params["loss"],
            focal_gamma=params["focal_gamma"],
            add_self_loops=True,
            device=device_pref,
        )

        print(f"\nTrial {idx} | params={params}")
        metrics = run_experiment(config, cached=cached_data)
        leaderboard.append({
            "output": str(output_path),
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
        print(f"{rank:02d}. tgatv2 {metric}={score:.4f} ({row['output']})")


if __name__ == "__main__":
    main()
