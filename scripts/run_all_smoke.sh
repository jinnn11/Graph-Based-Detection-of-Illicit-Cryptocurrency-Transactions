#!/usr/bin/env bash
set -euo pipefail

# Smoke test: quick runs to verify pipelines end-to-end.
# Adjust DEVICE to cuda/cpu/mps as needed.
DEVICE=${DEVICE:-auto}

python src/data/feature_engineering.py \
  --data-dir data/raw \
  --add-degree \
  --output data/processed/tabular_features.csv

python src/data/build_graph.py \
  --data-dir data/raw \
  --features-path data/processed/tabular_features.csv \
  --output data/processed/graph_feat.pt \
  --add-reverse-edges \
  --normalize-features

python src/data/build_temporal_data.py \
  --data-dir data/raw \
  --features-path data/processed/tabular_features.csv \
  --output data/processed/temporal_data.pt

# TGATv2 (temporal GAT) quick run
python src/training/train_tgat.py \
  --data-dir data/raw \
  --graph-path data/processed/graph_feat.pt \
  --device "$DEVICE" \
  --add-self-loops \
  --epochs 2 --patience 1 \
  --output experiments/tgat_smoke_results.json

# Graph Transformer quick run
python src/training/train_graph_transformer.py \
  --data-dir data/raw \
  --graph-path data/processed/graph_feat.pt \
  --device "$DEVICE" \
  --add-self-loops \
  --epochs 2 --patience 1 \
  --output experiments/graph_transformer_smoke_results.json

# TGN quick run
python src/training/train_tgn.py \
  --temporal-path data/processed/temporal_data.pt \
  --device "$DEVICE" \
  --epochs 2 --patience 1 \
  --batch-size 1000 \
  --output experiments/tgn_smoke_results.json

echo "Smoke run complete."
