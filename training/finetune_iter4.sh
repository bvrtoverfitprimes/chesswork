#!/usr/bin/env bash
# Iteration 4 of the mine -> fine-tune -> A/B loop.
#
# Key difference vs the FAILED iter3 (42.9% vs production over 28 games): the
# aux set here is ONLY real mined blunders from actual games -- the synthetic
# material-imbalance set is dropped. iter1 (94%) and iter2 (68%) both used pure
# real-blunder data; iter3 added 356 synthetic bare-king/army-minus-piece
# positions and lost. That composition change is the prime suspect, so revert
# to the recipe that won twice, now with the v2 (before+after pair) data from
# SF@2400 games and any blunders mined from the rigorous-benchmark games.
#
# Pure-WDL loss, warm-start from the current production checkpoint, judged by
# A/B games (20-30 minimum) before any adoption. Production files untouched.
set -e
cd "$(dirname "$0")/.."
PY="training/.venv/Scripts/python.exe"

AUX_REPEAT="${1:-10}"
BASE_ROWS="${2:-1000000}"
INIT_FROM="${3:-training/checkpoints/blunder_ft2.pt}"
OUT_TAG="${4:-iter4_ft}"

echo "[1/4] encoding real-blunder aux set (NO synthetic material-imbalance data) ..."
"$PY" training/data/build_dataset.py \
  --input training/data/raw/blunder_labeled.jsonl \
          training/data/raw/blunder_labeled_1900.jsonl \
          training/data/raw/blunder_labeled_iter3_sf2100.jsonl \
          training/data/raw/blunder_labeled_v2_sf2400.jsonl \
          training/data/raw/overnight_blunders.jsonl \
          training/data/raw/benchmark_blunders_labeled.jsonl \
  --out training/data/processed/iter4_aux_threats.npz --val-fraction 0.05

echo "[2/4] building ${BASE_ROWS}-row base anchor ..."
"$PY" training/data/subset_dataset.py --in training/data/processed/threats_dataset.npz \
  --out training/data/processed/base_anchor.npz --rows "$BASE_ROWS"

echo "[3/4] fine-tuning (pure WDL, warm-start from ${INIT_FROM}, aux x${AUX_REPEAT}) ..."
"$PY" training/model/train.py \
  --dataset training/data/processed/base_anchor.npz \
  --aux-dataset training/data/processed/iter4_aux_threats.npz \
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

echo "DONE. A/B before adopting (20-30 games minimum -- 8 is NOT enough, see iter3):"
echo "  $PY tools/ab_weights.py --a engine/limit/nnue_${OUT_TAG}.bin --b engine/limit/nnue_weights.bin --games 24 --ms 500"
