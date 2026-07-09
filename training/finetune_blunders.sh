#!/usr/bin/env bash
# Fine-tune the current NNUE on mined weak-point (blunder) positions.
# Pure-WDL loss (no magnitude tricks — see WORKLOG §28), warm-started from current
# weights, small base anchor + heavily-oversampled blunders, low LR, few epochs.
# Judge by A/B games before adopting.
set -e
cd "$(dirname "$0")/.."
PY="training/.venv/Scripts/python.exe"

AUX_REPEAT="${1:-40}"
BASE_ROWS="${2:-1000000}"

echo "[1/5] combining mined blunder sets ..."
cat training/data/raw/blunder_labeled*.jsonl > training/data/raw/blunder_all.jsonl
wc -l training/data/raw/blunder_all.jsonl

echo "[2/5] encoding blunder positions (threats features) ..."
"$PY" training/data/build_dataset.py --input training/data/raw/blunder_all.jsonl \
  --out training/data/processed/blunder_threats.npz --val-fraction 0.05

echo "[3/5] building ${BASE_ROWS}-row base anchor ..."
"$PY" training/data/subset_dataset.py --in training/data/processed/threats_dataset.npz \
  --out training/data/processed/base_anchor.npz --rows "$BASE_ROWS"

echo "[4/5] fine-tuning (pure WDL, warm-start, aux x${AUX_REPEAT}) ..."
"$PY" training/model/train.py \
  --dataset training/data/processed/base_anchor.npz \
  --aux-dataset training/data/processed/blunder_threats.npz \
  --aux-repeat "$AUX_REPEAT" \
  --hidden 1024 \
  --init-from training/checkpoints/threats_model.pt \
  --mse-weight 0.0 \
  --lr 2e-4 --lr-decay-epoch 3 --lr-decay-factor 0.5 \
  --batch-size 8192 \
  --epochs 4 --early-stop-patience 4 --max-minutes 60 \
  --out training/checkpoints/blunder_ft.pt

echo "[5/5] exporting to test weights ..."
"$PY" training/model/export.py --checkpoint training/checkpoints/blunder_ft.pt \
  --out engine/human_limit/nnue_blunder_ft.bin

echo "DONE. A/B before adopting:"
echo "  $PY tools/ab_weights.py --a engine/human_limit/nnue_blunder_ft.bin --b engine/human_limit/nnue_weights.bin --games 30 --ms 500"
