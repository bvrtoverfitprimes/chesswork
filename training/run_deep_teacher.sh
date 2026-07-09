#!/usr/bin/env bash
# Large eval-training pass with a DEEPER Stockfish teacher (the principled lever).
# Designed from the lessons in WORKLOG §28: pure-WDL loss (proven for strength),
# NO magnitude/MSE tricks, NO heavy decisive oversampling. Just better ground-truth
# labels + adequate training, then judged by A/B games (not val metrics).
#
# Usage: bash training/run_deep_teacher.sh <depth> [num_positions]
#   depth          Stockfish label depth (8-10 recommended; deeper = better target, slower)
#   num_positions  optional cap on how many lichess positions to relabel (for faster iteration)
set -e
cd "$(dirname "$0")/.."
PY="training/.venv/Scripts/python.exe"
DEPTH="${1:-8}"
CAP="${2:-0}"
TAG="deep${DEPTH}"

RAW_IN="training/data/raw/lichess_6m.jsonl"
if [ "$CAP" -gt 0 ]; then
  RAW_IN="training/data/raw/lichess_${CAP}_subset.jsonl"
  head -n "$CAP" training/data/raw/lichess_6m.jsonl > "$RAW_IN"
fi

echo "[1/4] relabeling $RAW_IN at depth=$DEPTH ..."
"$PY" training/data/relabel_shallow.py --input "$RAW_IN" \
  --depth "$DEPTH" --workers 6 \
  --out "training/data/raw/${TAG}_labeled.jsonl"

echo "[2/4] building dataset (threats features) ..."
"$PY" training/data/build_dataset.py \
  --input "training/data/raw/${TAG}_labeled.jsonl" \
  --out "training/data/processed/${TAG}_dataset.npz"

echo "[3/4] training pure-WDL, warm-started from current best ..."
"$PY" training/model/train.py \
  --dataset "training/data/processed/${TAG}_dataset.npz" \
  --hidden 1024 \
  --init-from training/checkpoints/threats_model.pt \
  --mse-weight 0.0 \
  --lr 5e-4 --lr-decay-epoch 4 --lr-decay-factor 0.4 \
  --batch-size 8192 \
  --epochs 10 --early-stop-patience 3 --max-minutes 240 \
  --out "training/checkpoints/${TAG}_model.pt"

echo "[4/4] exporting to a test weights file (A/B before adopting) ..."
"$PY" training/model/export.py \
  --checkpoint "training/checkpoints/${TAG}_model.pt" \
  --out "engine/human_limit/nnue_${TAG}.bin"

echo "DONE. A/B it before adopting:"
echo "  $PY tools/ab_weights.py --a engine/human_limit/nnue_${TAG}.bin --b engine/human_limit/nnue_weights.bin --games 30 --ms 800"
