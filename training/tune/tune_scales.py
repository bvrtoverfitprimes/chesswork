"""Coordinate-descent Texel tuning of the 12 category-scale multipliers.
Uses raw_eval_cli 'batch' mode (reads FENs on stdin, RAW_TUNE env sets scales,
prints white-relative cp per line). Minimizes logistic loss of sigmoid(eval/K)
vs white-POV game result. Fast: whole dataset evaluated per scale trial.
"""
import math
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(REPO, "tools", "raw_eval_tune.exe")
POS = os.path.join(REPO, "training", "tune", "tuning_positions.txt")
NCAT = 12
NAMES = ["mob", "bmob", "kingA", "threats", "passed", "pawnStruct",
         "pieceQ", "rookF", "bishopPair", "endgameK", "center", "outposts"]


def load():
    fens, results = [], []
    for line in open(POS):
        line = line.rstrip("\n")
        if not line:
            continue
        fen, res = line.split("\t")
        fens.append(fen)
        results.append(float(res))
    return fens, results


def batch_eval(fens, scales):
    env = dict(os.environ)
    env["RAW_TUNE"] = " ".join(str(x) for x in scales)
    env["RAW_WEIGHT"] = "1"
    p = subprocess.run([CLI, "batch"], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, env=env)
    return [int(x) for x in p.stdout.split()]


def loss(evals, results, K):
    s = 0.0
    for e, r in zip(evals, results):
        pr = 1.0 / (1.0 + 10 ** (-(e * K) / 400.0))
        s += (r - pr) ** 2
    return s / len(results)


def fit_K(evals, results):
    best, bestK = 1e9, 1.0
    for K in [x / 100 for x in range(20, 200, 5)]:
        l = loss(evals, results, K)
        if l < best:
            best, bestK = l, K
    return bestK, best


def main():
    fens, results = load()
    print(f"{len(fens)} positions", flush=True)
    scales = [128] * NCAT
    evals = batch_eval(fens, scales)
    K, base = fit_K(evals, results)
    print(f"baseline (identity scales) K={K:.2f} loss={base:.5f}", flush=True)

    improved = True
    rounds = 0
    while improved and rounds < 8:
        improved = False
        rounds += 1
        for i in range(NCAT):
            best_v, best_l = scales[i], loss(batch_eval(fens, scales), results, K)
            for v in [scales[i] + d for d in (-48, -24, -12, 12, 24, 48, 96)]:
                if v < 0 or v > 400:
                    continue
                trial = scales[:]
                trial[i] = v
                ev = batch_eval(fens, trial)
                Ki, l = fit_K(ev, results)
                if l < best_l - 1e-6:
                    best_l, best_v = l, v
            if best_v != scales[i]:
                scales[i] = best_v
                improved = True
                print(f"  round {rounds}: {NAMES[i]} -> {best_v} (loss {best_l:.5f})", flush=True)
        # refit K
        K, _ = fit_K(batch_eval(fens, scales), results)

    final = loss(batch_eval(fens, scales), results, K)
    print(f"\nFINAL scales: {scales}")
    print(f"loss {base:.5f} -> {final:.5f} (K={K:.2f})")
    print("RAW_TUNE=\"" + " ".join(str(x) for x in scales) + "\"")


if __name__ == "__main__":
    main()
