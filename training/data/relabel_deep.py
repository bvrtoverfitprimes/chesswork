"""Deep-teacher regrade of a sampled subset of the corpus (resumable).

Same shape as relabel_shallow.py, plus: --sample N (random subset sized to the
overnight budget) and resume (skips fens already present in --out from a
previous interrupted run; appends instead of overwriting).
"""
import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["training/data/raw/shallow4_labeled.jsonl"])
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--sample", type=int, default=0, help="0 = all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=0, help="0 = no wall-clock cap")
    parser.add_argument("--out", default="training/data/raw/deep12_labeled.jsonl")
    args = parser.parse_args()

    fens = []
    seen = set()
    for pattern in args.input:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fen = json.loads(line)["fen"]
                    if fen not in seen:
                        seen.add(fen)
                        fens.append(fen)

    if args.sample and args.sample < len(fens):
        rng = random.Random(args.seed)
        fens = rng.sample(fens, args.sample)

    done_fens = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_fens.add(json.loads(line)["fen"])
        fens = [f for f in fens if f not in done_fens]
        print(f"resume: {len(done_fens)} already labeled, {len(fens)} remaining", flush=True)

    print(f"relabeling {len(fens)} fens at depth={args.depth}", flush=True)
    t0 = time.time()
    done = 0
    with open(args.out, "a") as out_f:
        for start in range(0, len(fens), args.chunk_size):
            if args.max_hours and (time.time() - t0) > args.max_hours * 3600:
                print(f"wall-clock cap {args.max_hours}h reached, stopping (resumable)", flush=True)
                break
            chunk = fens[start:start + args.chunk_size]
            results = label_positions_parallel(chunk, args.stockfish, args.depth, args.workers)
            for r in results:
                out_f.write(json.dumps(r) + "\n")
            out_f.flush()
            done += len(chunk)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(fens) - done) / rate if rate > 0 else float("inf")
            print(f"{done}/{len(fens)} ({rate:.0f}/s, eta {eta/3600:.1f}h)", flush=True)

    print(f"done: {done} labeled -> {args.out} in {(time.time() - t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
