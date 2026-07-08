import argparse
import json
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", default="training/data/raw/lichess_400k.jsonl")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--eval-cli", default="tools/eval_cli")
    parser.add_argument("--weights", default="engine/human_limit/nnue_weights.bin")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rows = []
    with open(args.positions) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False)

    preds, targets = [], []
    for i in idx:
        row = rows[i]
        if row.get("cp") is None:
            continue
        out = subprocess.run([args.eval_cli, row["fen"], args.weights], capture_output=True, text=True)
        if out.returncode != 0:
            continue
        preds.append(float(out.stdout.strip()))
        targets.append(float(row["cp"]))

    preds = np.array(preds)
    targets = np.array(targets)
    targets_clamped = np.clip(targets, -3000, 3000)
    corr = float(np.corrcoef(preds, targets)[0, 1])
    corr_clamped = float(np.corrcoef(preds, targets_clamped)[0, 1])
    mae = float(np.mean(np.abs(preds - targets)))
    print(f"n={len(preds)} pearson_r={corr:.4f} pearson_r_clamped={corr_clamped:.4f} mae_cp={mae:.1f}")
    print(f"target cp range: min={targets.min():.0f} max={targets.max():.0f}")


if __name__ == "__main__":
    main()
