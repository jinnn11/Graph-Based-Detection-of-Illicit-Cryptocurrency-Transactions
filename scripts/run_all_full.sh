#!/usr/bin/env bash
set -euo pipefail

# Full run for 4090. Adjust MAX_TRIALS / EPOCHS if needed.
DEVICE=${DEVICE:-cuda}

# 1) Feature engineering
python src/data/feature_engineering.py \
  --data-dir data/raw \
  --add-degree \
  --output data/processed/tabular_features.csv

# 2) Build graph with engineered features
python src/data/build_graph.py \
  --data-dir data/raw \
  --features-path data/processed/tabular_features.csv \
  --output data/processed/graph_feat.pt \
  --add-reverse-edges \
  --normalize-features

# 3) Build temporal data for TGN
python src/data/build_temporal_data.py \
  --data-dir data/raw \
  --features-path data/processed/tabular_features.csv \
  --output data/processed/temporal_data.pt

# 4) XGBoost tuned baseline (save preds)
python src/models/tabular.py \
  --data-dir data/raw \
  --features-path data/processed/tabular_features.csv \
  --max-trials 30 \
  --early-stopping-rounds 50 \
  --output experiments/baseline_results.json \
  --save-preds experiments/xgb_preds.npz

# 5) TGATv2 sweep (temporal attention)
python src/training/sweep_tgat.py \
  --data-dir data/raw \
  --graph-path data/processed/graph_feat.pt \
  --device "$DEVICE" \
  --epochs 120 --patience 15 --max-trials 12 \
  --output-dir experiments/tgat_sweep

# 6) Focused TGATv2 sweep around best prior trial
if [ -f experiments/tgat_sweep/summary.json ]; then
  BEST_BASE=$(python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path('experiments/tgat_sweep/summary.json').read_text())
print(summary[0]['output'] if summary else '')
PY
)
  if [ -n "$BEST_BASE" ]; then
    python src/training/sweep_tgat_focus.py \
      --base "$BEST_BASE" \
      --data-dir data/raw \
      --graph-path data/processed/graph_feat.pt \
      --device "$DEVICE" \
      --epochs 120 --patience 15 --max-trials 12 \
      --output-dir experiments/tgat_focus
  fi
fi

# 7) Graph Transformer baseline (save preds)
python src/training/train_graph_transformer.py \
  --data-dir data/raw \
  --graph-path data/processed/graph_feat.pt \
  --device "$DEVICE" \
  --add-self-loops \
  --epochs 120 --patience 15 \
  --output experiments/graph_transformer_results.json \
  --save-preds experiments/graph_transformer_preds.npz

# 8) TGN baseline
python src/training/train_tgn.py \
  --temporal-path data/processed/temporal_data.pt \
  --device "$DEVICE" \
  --epochs 50 --patience 10 \
  --batch-size 2000 \
  --output experiments/tgn_results.json

# 9) Ensemble (XGB + best TGAT if preds available)
if [ -f experiments/xgb_preds.npz ] && [ -f experiments/tgat_preds.npz ]; then
  python src/evaluation/ensemble.py \
    --preds-a experiments/xgb_preds.npz \
    --preds-b experiments/tgat_preds.npz \
    --weight-a 0.95 \
    --output experiments/ensemble_results.json
fi

# 10) Leaderboard
python src/evaluation/leaderboard.py --metric pr_auc --top-k 20

echo "Full run complete."
