"""Sample positions from played games (rigorous_benchmark / overnight format),
grade with Stockfish, and attach each game's real outcome (white POV) so
training can blend eval labels with game results (nnue.md lambda blending).

Output rows: {fen, cp, mate, depth, result_w}
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", required=True)
    p.add_argument("--sample-every", type=int, default=3)
    p.add_argument("--skip-first-plies", type=int, default=8)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fen_result = {}
    n_games = 0
    for path in args.games:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                n_games += 1
                result_w = g["result"] if g["we_white"] else 1.0 - g["result"]
                for i, (fen, _we) in enumerate(g["plies"]):
                    if i < args.skip_first_plies or i % args.sample_every:
                        continue
                    fen_result.setdefault(fen, result_w)

    fens = list(fen_result.keys())
    print(f"{n_games} games -> {len(fens)} sampled positions; grading at depth {args.depth}...",
          flush=True)
    labeled = label_positions_parallel(fens, DEFAULT_STOCKFISH_PATH, args.depth, args.workers)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in labeled:
            r["result_w"] = fen_result[r["fen"]]
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(labeled)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
