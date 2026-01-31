"""Train a TGN baseline on temporal data."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_curve

try:
    from torch_geometric.data import TemporalData
    from torch_geometric.loader import TemporalDataLoader
    try:
        from torch_geometric.nn.models.tgn import TGN
        TGN_AVAILABLE = True
    except ImportError:
        from torch_geometric.nn.models import TGN
        TGN_AVAILABLE = True
except ImportError:
    TGN_AVAILABLE = False
    try:
        from torch_geometric.data import TemporalData
        from torch_geometric.loader import TemporalDataLoader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch_geometric with temporal support is required for TGN."
        ) from exc
    try:
        from torch_geometric.nn.models import tgn as tgn_module
        TGNMemory = tgn_module.TGNMemory
        IdentityMessage = tgn_module.IdentityMessage
        LastAggregator = tgn_module.LastAggregator
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "TGN not available. Install a PyG version that includes TGN or a "
            "torch-geometric-temporal build that exposes TGN."
        ) from exc


@dataclass
class TrainConfig:
    temporal_path: str
    epochs: int
    hidden_dim: int
    memory_dim: int
    time_dim: int
    message_dim: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    output: str
    patience: int
    device: str | None = None


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
    parser = argparse.ArgumentParser(description="TGN baseline")
    parser.add_argument("--temporal-path", default="data/processed/temporal_data.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--memory-dim", type=int, default=128)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--message-dim", type=int, default=166)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", default="experiments/tgn_results.json")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Device selection (auto prefers cuda then mps)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device_pref = None if args.device == "auto" else args.device
    device = get_device(device_pref)
    print(f"Using device: {device}")

    payload = torch.load(args.temporal_path, map_location="cpu")
    data = TemporalData(
        src=payload["src"],
        dst=payload["dst"],
        t=payload["t"].float(),
        msg=payload["msg"],
    )
    data = data.to(device)

    y = payload.get("y")
    if y is None:
        # Build labels if provided in payload
        raise RuntimeError("Labels not found in temporal payload. Add labels before training.")

    y = y.to(device)
    train_mask = payload["train_mask"].to(device)
    val_mask = payload["val_mask"].to(device)
    test_mask = payload["test_mask"].to(device)

    num_nodes = payload["x"].size(0)

    loader = TemporalDataLoader(data, batch_size=args.batch_size)

    if TGN_AVAILABLE:
        tgn = TGN(
            num_nodes=num_nodes,
            raw_msg_dim=payload["msg"].size(1),
            memory_dim=args.memory_dim,
            time_dim=args.time_dim,
            embedding_dim=args.hidden_dim,
        ).to(device)

        decoder = torch.nn.Linear(args.hidden_dim, 2).to(device)
        optimizer = torch.optim.Adam(
            list(tgn.parameters()) + list(decoder.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        # Minimal TGN-style memory pipeline using PyG building blocks
        memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=payload["msg"].size(1),
            memory_dim=args.memory_dim,
            time_dim=args.time_dim,
            message_module=IdentityMessage(payload["msg"].size(1), args.memory_dim, args.time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)
        decoder = torch.nn.Sequential(
            torch.nn.Linear(args.memory_dim, args.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(args.hidden_dim, 2),
        ).to(device)
        optimizer = torch.optim.Adam(
            list(memory.parameters()) + list(decoder.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    best_val = -np.inf
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        if TGN_AVAILABLE:
            tgn.train()
            tgn.reset_state()
        else:
            memory.train()
            memory.reset_state()
        decoder.train()

        losses = []
        for batch in loader:
            optimizer.zero_grad()
            nodes = torch.cat([batch.src, batch.dst])
            labels = y[nodes]
            labeled = labels >= 0
            if labeled.sum().item() == 0:
                continue
            if TGN_AVAILABLE:
                z, _ = tgn(batch.src, batch.dst, batch.t, batch.msg)
                logits = decoder(z)[labeled]
            else:
                memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
                mem, _ = memory(nodes)
                logits = decoder(mem)[labeled]
            loss = F.cross_entropy(logits, labels[labeled])
            loss.backward()
            optimizer.step()
            if TGN_AVAILABLE:
                if hasattr(tgn, "detach"):
                    tgn.detach()
            else:
                if hasattr(memory, "detach"):
                    memory.detach()
            losses.append(float(loss.item()))

        # validation pass (full graph embeddings)
        if TGN_AVAILABLE:
            tgn.eval()
            tgn.reset_state()
            zs = torch.zeros((num_nodes, args.hidden_dim), device=device)
        else:
            memory.eval()
            memory.reset_state()
            zs = torch.zeros((num_nodes, args.memory_dim), device=device)

        decoder.eval()
        for batch in loader:
            if TGN_AVAILABLE:
                z, _ = tgn(batch.src, batch.dst, batch.t, batch.msg)
                nodes = torch.cat([batch.src, batch.dst])
                zs[nodes] = z.detach()
            else:
                memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
                nodes = torch.cat([batch.src, batch.dst])
                mem, _ = memory(nodes)
                zs[nodes] = mem.detach()

        with torch.no_grad():
            logits = decoder(zs)
            val_scores = torch.softmax(logits[val_mask], dim=1)[:, 1].detach().cpu().numpy()
            val_metrics = eval_scores(y[val_mask].detach().cpu().numpy(), val_scores)

        print(f"Epoch {epoch:03d} | loss={np.mean(losses):.4f} | val_pr_auc={val_metrics['pr_auc']:.4f}")

        if val_metrics["pr_auc"] > best_val:
            best_val = val_metrics["pr_auc"]
            if TGN_AVAILABLE:
                best_state = {
                    "tgn": {k: v.detach().cpu() for k, v in tgn.state_dict().items()},
                    "decoder": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                }
            else:
                best_state = {
                    "memory": {k: v.detach().cpu() for k, v in memory.state_dict().items()},
                    "decoder": {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # test
    if best_state is not None:
        if TGN_AVAILABLE:
            tgn.load_state_dict(best_state["tgn"])
        else:
            memory.load_state_dict(best_state["memory"])
        decoder.load_state_dict(best_state["decoder"])

    if TGN_AVAILABLE:
        tgn.eval()
        tgn.reset_state()
        zs = torch.zeros((num_nodes, args.hidden_dim), device=device)
    else:
        memory.eval()
        memory.reset_state()
        zs = torch.zeros((num_nodes, args.memory_dim), device=device)

    decoder.eval()
    for batch in loader:
        if TGN_AVAILABLE:
            z, _ = tgn(batch.src, batch.dst, batch.t, batch.msg)
            nodes = torch.cat([batch.src, batch.dst])
            zs[nodes] = z.detach()
        else:
            memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)
            nodes = torch.cat([batch.src, batch.dst])
            mem, _ = memory(nodes)
            zs[nodes] = mem.detach()

    with torch.no_grad():
        logits = decoder(zs)
        test_scores = torch.softmax(logits[test_mask], dim=1)[:, 1].detach().cpu().numpy()
        test_metrics = eval_scores(y[test_mask].detach().cpu().numpy(), test_scores)

    print("Test metrics:", test_metrics)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "metrics": test_metrics,
        "val_metrics": val_metrics,
        "config": {
            "model": "tgn",
            "tgn_available": TGN_AVAILABLE,
            "hidden_dim": args.hidden_dim,
            "memory_dim": args.memory_dim,
            "time_dim": args.time_dim,
            "message_dim": args.message_dim,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
    }
    Path(args.output).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
