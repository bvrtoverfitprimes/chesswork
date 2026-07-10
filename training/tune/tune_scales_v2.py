"""Well-conditioned category-scale tuning: numpy-vectorized, tighter bounds,
L2 regularization toward identity (128) to prevent overfit bound-hitting.
Caches per-category eval CONTRIBUTIONS once (via RAW_TUNE unit-probing) so the
whole search is pure numpy -- no per-trial subprocess. Requires raw_eval_tune
'batch' mode + the 12 category scales."""
import os
import subprocess
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.path.join(REPO, "tools", "raw_eval_tune.exe")
POS = os.path.join(REPO, "training", "tune", "tuning_positions.txt")
NCAT = 12
NAMES = ["mob", "bmob", "kingA", "threats", "passed", "pawnStruct",
         "pieceQ", "rookF", "bishopPair", "endgameK", "center", "outposts"]


def batch_eval(fens, scales):
    env = dict(os.environ); env["RAW_WEIGHT"] = "1"
    env["RAW_TUNE"] = " ".join(str(int(x)) for x in scales)
    p = subprocess.run([CLI, "batch"], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, env=env)
    return np.array([int(x) for x in p.stdout.split()], dtype=np.float64)


def main():
    fens, res = [], []
    for line in open(POS):
        line = line.rstrip("\n")
        if line:
            f, r = line.split("\t"); fens.append(f); res.append(float(r))
    res = np.array(res)
    print(f"{len(fens)} positions", flush=True)

    # Decompose eval into base (all cats at 0) + per-category contribution at 128.
    # eval(scales) = base + sum_i (scales_i/128) * contrib_i  (exact: eval is linear in each scale)
    zero = [0] * NCAT
    base = batch_eval(fens, zero)
    contrib = np.zeros((len(fens), NCAT))
    for i in range(NCAT):
        s = zero[:]; s[i] = 128
        contrib[:, i] = batch_eval(fens, s) - base
    print("decomposed eval into base + 12 linear category contributions", flush=True)

    def evals(scales):
        return base + contrib @ (np.array(scales) / 128.0)

    def loss(scales, K, lam=0.0):
        e = evals(scales)
        pr = 1.0 / (1.0 + np.power(10.0, -(e * K) / 400.0))
        mse = np.mean((res - pr) ** 2)
        reg = lam * np.mean((np.array(scales) - 128.0) ** 2) / (128.0 ** 2)
        return mse + reg

    def best_K(scales):
        e = evals(scales)
        bestK, bl = 1.0, 1e9
        for K in np.arange(0.15, 1.2, 0.02):
            pr = 1.0 / (1.0 + np.power(10.0, -(e * K) / 400.0))
            l = np.mean((res - pr) ** 2)
            if l < bl:
                bl, bestK = l, K
        return bestK, bl

    scales = [128.0] * NCAT
    K, base_loss = best_K(scales)
    print(f"identity: K={K:.2f} mse={base_loss:.5f}", flush=True)
    LAM = 0.02  # regularization strength
    LO, HI = 56, 220
    for rnd in range(12):
        moved = False
        for i in range(NCAT):
            cur = loss(scales, K, LAM)
            bestv, bestl = scales[i], cur
            for v in np.arange(LO, HI + 1, 8):
                t = scales[:]; t[i] = v
                l = loss(t, K, LAM)
                if l < bestl - 1e-7:
                    bestl, bestv = l, v
            if bestv != scales[i]:
                scales[i] = bestv; moved = True
        K, _ = best_K(scales)
        if not moved:
            break
    final_mse = best_K(scales)[1]
    print(f"\nregularized tune (bounds [{LO},{HI}], L2={LAM}):")
    for i in range(NCAT):
        print(f"  {NAMES[i]:12s} {int(scales[i])}")
    print(f"mse {base_loss:.5f} -> {final_mse:.5f}  K={K:.2f}")
    print("RAW_TUNE=\"" + " ".join(str(int(x)) for x in scales) + "\"")


if __name__ == "__main__":
    main()
