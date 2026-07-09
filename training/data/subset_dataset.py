"""Slice a processed CSR dataset's TRAIN split down to the first N rows (rows are
already shuffled at build time, so the prefix is a random sample). Val split is
kept as-is. Used to build a smaller base anchor for fast weak-point fine-tuning."""
import argparse

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--rows", type=int, default=1_000_000)
    args = p.parse_args()

    z = np.load(args.inp)
    n_train = len(z["train_targets"])
    n = min(args.rows, n_train)

    def slice_split(prefix, n):
        off = z[f"{prefix}_stm_off"]
        noff = z[f"{prefix}_ntm_off"]
        return {
            f"{prefix}_stm_idx": z[f"{prefix}_stm_idx"][:off[n]],
            f"{prefix}_stm_off": off[:n + 1],
            f"{prefix}_ntm_idx": z[f"{prefix}_ntm_idx"][:noff[n]],
            f"{prefix}_ntm_off": noff[:n + 1],
            f"{prefix}_targets": z[f"{prefix}_targets"][:n],
            f"{prefix}_buckets": z[f"{prefix}_buckets"][:n],
        }

    out = {}
    out.update(slice_split("train", n))
    # keep full val
    for k in ("val_stm_idx", "val_stm_off", "val_ntm_idx", "val_ntm_off", "val_targets", "val_buckets"):
        out[k] = z[k]

    np.savez(args.out, **out)
    print(f"subset {n}/{n_train} train rows (+{len(z['val_targets'])} val) -> {args.out}")


if __name__ == "__main__":
    main()
