"""Baseline tabular model (no graph) for Elliptic AML dataset."""
from __future__ import annotations

import argparse
import json
import random
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "xgboost is required for the tabular baseline. "
        "Install with `pip install xgboost`."
    ) from exc


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_features(features_path: Path) -> pd.DataFrame:
    features = pd.read_csv(features_path, header=None)
    n_cols = features.shape[1]
    feature_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(n_cols - 2)]
    features.columns = feature_cols
    return features


def load_labels(classes_path: Path) -> pd.DataFrame:
    return pd.read_csv(classes_path)


def map_labels(classes: pd.DataFrame) -> pd.DataFrame:
    class_map = {1: 1, 2: 0, "1": 1, "2": 0}
    classes = classes.copy()
    classes["label"] = classes["class"].map(class_map)
    return classes


def temporal_masks(time_step: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_mask = (time_step <= 34).to_numpy()
    val_mask = ((time_step >= 35) & (time_step <= 41)).to_numpy()
    test_mask = (time_step >= 42).to_numpy()
    return train_mask, val_mask, test_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def prepare_data(data_dir: Path) -> SplitData:
    features = load_features(data_dir / "elliptic_txs_features.csv")
    classes = load_labels(data_dir / "elliptic_txs_classes.csv")
    classes = map_labels(classes)

    data = features.merge(classes[["txId", "label"]], on="txId", how="left")
    labeled = data.dropna(subset=["label"]).copy()

    x = labeled.drop(columns=["txId", "time_step", "label"]).to_numpy()
    y = labeled["label"].astype(int).to_numpy()

    train_mask, val_mask, test_mask = temporal_masks(labeled["time_step"])

    return SplitData(
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_val=x[val_mask],
        y_val=y[val_mask],
        x_test=x[test_mask],
        y_test=y[test_mask],
    )


def compute_class_weight(y: np.ndarray) -> float:
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return neg / max(pos, 1)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.01) -> float:
    if len(y_true) == 0:
        return 0.0
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].mean())


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float = 0.01) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    valid = fpr <= fpr_target
    if not np.any(valid):
        return 0.0
    return float(tpr[valid].max())


def build_param_grid() -> list[dict]:
    grid = {
        "n_estimators": [300, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 5.0],
    }
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def train_with_params(
    split: SplitData,
    params: Dict[str, float],
    random_state: int,
    early_stopping_rounds: int,
) -> XGBClassifier:
    scale_pos_weight = compute_class_weight(split.y_train)
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        **params,
    )
    try:
        model.fit(
            split.x_train,
            split.y_train,
            eval_set=[(split.x_val, split.y_val)],
            verbose=False,
            early_stopping_rounds=early_stopping_rounds,
        )
    except TypeError:
        # Older XGBoost sklearn API may not support early_stopping_rounds
        print("  note: early_stopping_rounds not supported by this XGBoost version; training without it")
        model.fit(
            split.x_train,
            split.y_train,
            eval_set=[(split.x_val, split.y_val)],
            verbose=False,
        )
    return model


def tune_model(
    split: SplitData,
    seed: int,
    max_trials: int,
    early_stopping_rounds: int,
) -> Tuple[XGBClassifier, Dict[str, float], Dict[str, float], int]:
    rng = np.random.default_rng(seed)
    grid = build_param_grid()
    if max_trials >= len(grid):
        trials = grid
    else:
        indices = rng.choice(len(grid), size=max_trials, replace=False)
        trials = [grid[i] for i in indices]

    best_model = None
    best_params = None
    best_metrics = None
    best_score = -np.inf

    n_trials = len(trials)
    print(f"Tuning XGBoost over {n_trials} trials...")
    for idx, params in enumerate(trials, start=1):
        print(f"[{idx}/{len(trials)}] params={params}")
        model = train_with_params(
            split,
            params=params,
            random_state=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        val_metrics = evaluate(model, split.x_val, split.y_val)
        score = val_metrics["pr_auc"]
        print(f"  val_pr_auc={score:.4f}")
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
            best_metrics = val_metrics

    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("Tuning failed to produce a valid model.")

    return best_model, best_params, best_metrics, n_trials


def evaluate(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    y_score = model.predict_proba(x)[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, y_score)),
        "recall_at_1pct_fpr": recall_at_fpr(y, y_score, fpr_target=0.01),
        "precision_at_1pct": precision_at_k(y, y_score, k_frac=0.01),
    }


def save_results(
    output_path: Path,
    metrics: Dict[str, float],
    seed: int,
    best_params: Dict[str, float],
    val_metrics: Dict[str, float],
    n_trials: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "metrics": metrics,
        "val_metrics": val_metrics,
        "best_params": best_params,
        "n_trials": n_trials,
    }
    output_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabular baseline (no graph)")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        default="experiments/baseline_results.json",
        help="Where to save results JSON",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=30,
        help="Max hyperparameter trials for tuning",
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Early stopping rounds on validation set",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    split = prepare_data(Path(args.data_dir))
    model, best_params, val_metrics, n_trials = tune_model(
        split,
        seed=args.seed,
        max_trials=args.max_trials,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    metrics = evaluate(model, split.x_test, split.y_test)

    print("Test metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    save_results(
        Path(args.output),
        metrics,
        seed=args.seed,
        best_params=best_params,
        val_metrics=val_metrics,
        n_trials=n_trials,
    )


if __name__ == "__main__":
    main()
