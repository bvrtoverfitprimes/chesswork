#!/usr/bin/env bash
# Iteration 3 of the mine -> fine-tune -> A/B loop, PLUS a targeted
# material-imbalance/magnitude-calibration set (see mine_material_imbalance.py
# and WORKLOG §24/§25/tactical_sanity.py -- the one remaining known eval defect:
# a color asymmetry on decisive material advantages, worst case bare-king
# "black up a queen" scoring -453cp against a <-500cp bar).
# Pure-WDL loss (no magnitude tricks), warm-started from the CURRENT production
# checkpoint, small base anchor + oversampled weak points. Judge by A/B games
# before adopting -- never touch production files directly.
set -e
cd "$(dirname "$0")/.."
PY="training/.venv/Scripts/python.exe"

AUX_REPEAT="${1:-10}"
BASE_ROWS="${2:-1000000}"
INIT_FROM="${3:-training/checkpoints/blunder_ft2.pt}"
OUT_TAG="${4:-iter3_ft}"

echo "[1/4] encoding combined weak-point set (blunders iter1+2+overnight+iter3, + material imbalance) ..."
"$PY" training/data/build_dataset.py \
  --input training/data/raw/blunder_labeled*.jsonl training/data/raw/overnight_blunders.jsonl \
          training/data/raw/material_imbalance_labeled.jsonl \
  --out training/data/processed/iter3_aux_threats.npz --val-fraction 0.05

echo "[2/4] building ${BASE_ROWS}-row base anchor ..."
"$PY" training/data/subset_dataset.py --in training/data/processed/threats_dataset.npz \
  --out training/data/processed/base_anchor.npz --rows "$BASE_ROWS"

echo "[3/4] fine-tuning (pure WDL, warm-start from ${INIT_FROM}, aux x${AUX_REPEAT}) ..."
"$PY" training/model/train.py \
  --dataset training/data/processed/base_anchor.npz \
  --aux-dataset training/data/processed/iter3_aux_threats.npz \
  --aux-repeat "$AUX_REPEAT" \
  --hidden 1024 \
  --init-from "$INIT_FROM" \
  --mse-weight 0.0 \
  --lr 2e-4 --lr-decay-epoch 3 --lr-decay-factor 0.5 \
  --batch-size 8192 \
  --epochs 4 --early-stop-patience 4 --max-minutes 60 \
  --out "training/checkpoints/${OUT_TAG}.pt"

echo "[4/4] exporting to test weights ..."
"$PY" training/model/export.py --checkpoint "training/checkpoints/${OUT_TAG}.pt" \
  --out "engine/limit/nnue_${OUT_TAG}.bin"

echo "DONE. A/B before adopting:"
echo "  $PY tools/ab_weights.py --a engine/limit/nnue_${OUT_TAG}.bin --b engine/limit/nnue_weights.bin --games 8 --ms 500"
