import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=["training/data/raw/lichess_6m.jsonl",
                                                         "training/data/raw/curriculum_labeled.jsonl"])
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=20000)
    parser.add_argument("--out", default="training/data/raw/shallow2_labeled.jsonl")
    args = parser.parse_args()

    fens = []
    for pattern in args.input:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fens.append(json.loads(line)["fen"])
    print(f"loaded {len(fens)} fens to relabel at depth={args.depth}", flush=True)

    t0 = time.time()
    done = 0
    with open(args.out, "w") as out_f:
        for start in range(0, len(fens), args.chunk_size):
            chunk = fens[start:start + args.chunk_size]
            results = label_positions_parallel(chunk, args.stockfish, args.depth, args.workers)
            for r in results:
                out_f.write(json.dumps(r) + "\n")
            out_f.flush()
            done += len(chunk)
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (len(fens) - done) / rate if rate > 0 else float("inf")
            print(f"{done}/{len(fens)} ({rate:.0f}/s, eta {eta/60:.1f}min)", flush=True)

    print(f"done: wrote to {args.out} in {(time.time() - t0)/60:.1f}min")


if __name__ == "__main__":
    main()
