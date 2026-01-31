"""Ensemble predictions from two models and evaluate metrics."""
from __future__ import annotations

import argparse
import numpy as np
from sklearn.metrics import average_precision_score, roc_curve


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.01) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = fpr <= fpr_target
    if not np.any(valid):
        return 0.0
    return float(tpr[valid].max())


def eval_scores(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_score, fpr_target=0.01),
        "precision_at_1pct": precision_at_k(y_true, y_score, k_frac=0.01),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble two prediction files")
    parser.add_argument("--preds-a", required=True, help="NPZ with y_val/y_test/score_val/score_test")
    parser.add_argument("--preds-b", required=True, help="NPZ with y_val/y_test/score_val/score_test")
    parser.add_argument("--weight-a", type=float, default=0.5, help="Weight for model A")
    args = parser.parse_args()

    a = np.load(args.preds_a)
    b = np.load(args.preds_b)

    w = args.weight_a
    score_val = w * a["score_val"] + (1 - w) * b["score_val"]
    score_test = w * a["score_test"] + (1 - w) * b["score_test"]

    val_metrics = eval_scores(a["y_val"], score_val)
    test_metrics = eval_scores(a["y_test"], score_test)

    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
