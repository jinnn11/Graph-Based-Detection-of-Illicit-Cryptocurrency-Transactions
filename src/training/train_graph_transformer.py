"""Train Graph Transformer (TransformerConv) baseline."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_curve

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.graph_transformer import build_graph_transformer


@dataclass
class GraphPayload:
    x: torch.Tensor
    edge_index: torch.Tensor
    time_step: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    id_mapping: Dict[int, int]
    x_mean: torch.Tensor | None = None
    x_std: torch.Tensor | None = None
    add_reverse_edges: bool = False
    normalized: bool = False


@dataclass
class TrainConfig:
    data_dir: str
    graph_path: str
    epochs: int
    hidden_dim: int
    num_layers: int
    dropout: float
    lr: float
    weight_decay: float
    heads: int
    seed: int
    output: str
    model_output: str
    patience: int
    loss: str
    focal_gamma: float
    add_self_loops: bool
    device: str | None = None
    save_preds: str | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred is not None:
        if preferred == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        if preferred == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_graph(graph_path: Path) -> GraphPayload:
    payload = torch.load(graph_path, map_location="cpu")
    return GraphPayload(**payload)


def load_labels(data_dir: Path) -> pd.DataFrame:
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    class_map = {1: 1, 2: 0, "1": 1, "2": 0}
    classes = classes.copy()
    classes["label"] = classes["class"].map(class_map)
    return classes[["txId", "label"]]


def build_label_tensor(id_mapping: Dict[int, int], labels: pd.DataFrame, num_nodes: int) -> torch.Tensor:
    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for raw_id, label in labels.dropna().itertuples(index=False):
        idx = id_mapping.get(int(raw_id))
        if idx is not None:
            y[idx] = int(label)
    return y


def build_labeled_masks(
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    labeled = y >= 0
    return train_mask & labeled, val_mask & labeled, test_mask & labeled


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    y_masked = y[mask]
    pos = (y_masked == 1).sum().item()
    neg = (y_masked == 0).sum().item()
    total = pos + neg
    if pos == 0 or neg == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    weight_neg = total / (2 * neg)
    weight_pos = total / (2 * pos)
    return torch.tensor([weight_neg, weight_pos], dtype=torch.float32)


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    idx = torch.arange(targets.size(0), device=targets.device)
    pt = probs[idx, targets]
    log_pt = log_probs[idx, targets]
    weights = class_weights[targets]
    loss = -weights * (1 - pt).pow(gamma) * log_pt
    return loss.mean()


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


def evaluate(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    if mask.sum().item() == 0:
        return {"pr_auc": 0.0, "recall_at_1pct_fpr": 0.0, "precision_at_1pct": 0.0}
    scores = torch.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
    y_true = y[mask].detach().cpu().numpy()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, scores, fpr_target=0.01),
        "precision_at_1pct": precision_at_k(y_true, scores, k_frac=0.01),
    }


def save_preds(path: str, y: torch.Tensor, logits: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor) -> None:
    val_scores = torch.softmax(logits[val_mask], dim=1)[:, 1].detach().cpu().numpy()
    test_scores = torch.softmax(logits[test_mask], dim=1)[:, 1].detach().cpu().numpy()
    np.savez(
        path,
        y_val=y[val_mask].detach().cpu().numpy(),
        y_test=y[test_mask].detach().cpu().numpy(),
        score_val=val_scores,
        score_test=test_scores,
    )


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    loss_fn,
) -> float:
    model.train()
    optimizer.zero_grad()
    logits = model(x, edge_index)
    loss = loss_fn(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph Transformer baseline")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--graph-path", default="data/processed/graph.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/graph_transformer_results.json")
    parser.add_argument("--model-output", default="experiments/graph_transformer.pt")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="cross_entropy")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--add-self-loops", action="store_true")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device selection (auto prefers cuda then mps)",
    )
    parser.add_argument(
        "--save-preds",
        default=None,
        help="Optional path to save npz with val/test scores",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device_pref = None if args.device == "auto" else args.device
    device = get_device(device_pref)
    print(f"Using device: {device}")

    graph = load_graph(Path(args.graph_path))
    labels = load_labels(Path(args.data_dir))
    y = build_label_tensor(graph.id_mapping, labels, num_nodes=graph.x.size(0))

    train_mask, val_mask, test_mask = build_labeled_masks(y, graph.train_mask, graph.val_mask, graph.test_mask)

    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if args.add_self_loops:
        from torch_geometric.utils import add_self_loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

    model = build_graph_transformer(
        in_channels=x.size(1),
        hidden_channels=args.hidden_dim,
        out_channels=2,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(y, train_mask).to(device)
    if args.loss == "focal":
        def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            return focal_loss(logits, targets, class_weights, gamma=args.focal_gamma)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -np.inf
    best_epoch = -1
    best_state = None
    best_val_metrics = None
    patience_counter = 0

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, optimizer, x, edge_index, y, train_mask, loss_fn)

        model.eval()
        with torch.no_grad():
            logits = model(x, edge_index)
            val_metrics = evaluate(logits, y, val_mask)

        if val_metrics["pr_auc"] > best_val:
            best_val = val_metrics["pr_auc"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={loss:.4f} | "
            f"val_pr_auc={val_metrics['pr_auc']:.4f}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (patience={args.patience})")
            break

    if best_state is not None:
        Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, args.model_output)
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        test_metrics = evaluate(logits, y, test_mask)

    print("Best epoch:", best_epoch)
    print("Test metrics:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

    if args.save_preds:
        save_preds(args.save_preds, y, logits, val_mask, test_mask)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "metrics": test_metrics,
        "val_metrics": best_val_metrics or {},
        "config": {
            "model": "graph_transformer",
            "heads": args.heads,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "loss": args.loss,
            "focal_gamma": args.focal_gamma,
            "add_self_loops": args.add_self_loops,
            "graph_normalized": graph.normalized,
            "graph_reverse_edges": graph.add_reverse_edges,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
