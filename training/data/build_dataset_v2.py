import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from encoding.encode import fen_to_feature_indices, fen_to_output_bucket

CP_SCALE = 400.0
CP_CLAMP = 3000.0
MATE_EQUIV_CP = 3000.0


def white_relative_cp(row):
    if row.get("cp") is not None:
        return float(row["cp"])
    mate = row["mate"]
    return MATE_EQUIV_CP if mate > 0 else -MATE_EQUIV_CP


def stm_relative_target(row, result_lambda=1.0):
    white_cp = white_relative_cp(row)
    turn = row["fen"].split(" ")[1]
    stm_cp = white_cp if turn == "w" else -white_cp
    stm_cp = max(-CP_CLAMP, min(CP_CLAMP, stm_cp))
    logit = stm_cp / CP_SCALE
    if result_lambda >= 1.0 or "result_w" not in row:
        return logit
    # blend eval-WDL with the real game outcome in WDL space, then back to logit
    import math
    result_stm = row["result_w"] if turn == "w" else 1.0 - row["result_w"]
    wdl = result_lambda * (1.0 / (1.0 + math.exp(-logit))) + (1.0 - result_lambda) * result_stm
    wdl = min(max(wdl, 1e-4), 1.0 - 1e-4)
    return math.log(wdl / (1.0 - wdl))


def build(input_globs, out_path, val_fraction, seed, result_lambda=1.0):
    rows = []
    for pattern in input_globs:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))

    rng = np.random.default_rng(seed)
    rng.shuffle(rows)

    stm_indices, stm_offsets = [], [0]
    ntm_indices, ntm_offsets = [], [0]
    targets = []
    buckets = []

    for row in rows:
        stm_idx, ntm_idx = fen_to_feature_indices(row["fen"])
        stm_indices.extend(stm_idx)
        stm_offsets.append(len(stm_indices))
        ntm_indices.extend(ntm_idx)
        ntm_offsets.append(len(ntm_indices))
        targets.append(stm_relative_target(row, result_lambda))
        buckets.append(fen_to_output_bucket(row["fen"]))

    n = len(targets)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    def slice_csr(indices, offsets, lo, hi):
        start, end = offsets[lo], offsets[hi]
        return np.array(indices[start:end], dtype=np.int32), np.array(offsets[lo:hi + 1], dtype=np.int64) - start

    train_stm_idx, train_stm_off = slice_csr(stm_indices, stm_offsets, 0, n_train)
    train_ntm_idx, train_ntm_off = slice_csr(ntm_indices, ntm_offsets, 0, n_train)
    val_stm_idx, val_stm_off = slice_csr(stm_indices, stm_offsets, n_train, n)
    val_ntm_idx, val_ntm_off = slice_csr(ntm_indices, ntm_offsets, n_train, n)

    targets = np.array(targets, dtype=np.float32)
    buckets = np.array(buckets, dtype=np.int8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        train_stm_idx=train_stm_idx, train_stm_off=train_stm_off,
        train_ntm_idx=train_ntm_idx, train_ntm_off=train_ntm_off,
        train_targets=targets[:n_train], train_buckets=buckets[:n_train],
        val_stm_idx=val_stm_idx, val_stm_off=val_stm_off,
        val_ntm_idx=val_ntm_idx, val_ntm_off=val_ntm_off,
        val_targets=targets[n_train:], val_buckets=buckets[n_train:],
    )
    print(f"built dataset: {n_train} train / {n - n_train} val -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["training/data/raw/*.jsonl"])
    parser.add_argument("--out", default="training/data/processed/dataset.npz")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-lambda", type=float, default=1.0,
                        help="1.0 = pure eval labels; <1 blends real game outcome for rows with result_w")
    args = parser.parse_args()
    build(args.input, args.out, args.val_fraction, args.seed, args.result_lambda)


if __name__ == "__main__":
    main()
