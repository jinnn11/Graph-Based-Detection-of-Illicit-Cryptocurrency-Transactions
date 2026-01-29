"""Leaderboard utility for experiment results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_results(path: Path) -> Dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "metrics" not in payload:
        return None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--metric", default="pr_auc")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    exp_dir = Path(args.experiments_dir)
    result_files = sorted(exp_dir.glob("**/*results.json"))
    result_files += sorted(exp_dir.glob("**/*tgat*.json"))

    rows: List[Dict[str, Any]] = []
    for path in result_files:
        payload = load_results(path)
        if payload is None:
            continue
        metrics = payload.get("metrics", {})
        if args.metric not in metrics:
            continue
        config = payload.get("config", {})
        model = config.get("model") or path.stem.replace("_results", "")
        rows.append({
            "path": path,
            "model": model,
            "metric": metrics[args.metric],
            "metrics": metrics,
            "config": config,
        })

    rows.sort(key=lambda x: x["metric"], reverse=True)

    print(f"Found {len(rows)} result files. Metric: {args.metric}")
    for rank, row in enumerate(rows[: args.top_k], start=1):
        metrics = row["metrics"]
        print(
            f"{rank:02d}. {row['model']} | "
            f"pr_auc={metrics.get('pr_auc', 0):.4f} | "
            f"recall@1%={metrics.get('recall_at_1pct_fpr', 0):.4f} | "
            f"precision@1%={metrics.get('precision_at_1pct', 0):.4f} | "
            f"file={row['path']}"
        )


if __name__ == "__main__":
    main()
