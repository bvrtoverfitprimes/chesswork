"""Mine weak-point training positions from ALREADY-PLAYED games (e.g. the game
trails saved by tools/rigorous_benchmark.py), instead of playing new ones.

Reuses mine_blunders_v2's detection (before+after pairs, before-ceiling filter,
depth-consistency diagnostic) and perturbation, so benchmark games double as
training-data collection -- no engine time wasted.

Input: jsonl, one game per line, each {plies: [[fen, we_move], ...],
we_white: bool, result: float} (extra keys like "opponent" are ignored).
Output: {fen, cp, mate, depth} jsonl ready for build_dataset.py.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mine_blunders_v2 import detect_blunders, perturb, our_advantage_cp
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH


def load_games(path):
    games = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            # plies may arrive as lists (json) rather than tuples -- normalize
            g["plies"] = [(fen, we) for fen, we in g["plies"]]
            games.append(g)
    return games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", default="training/data/raw/rigorous_benchmark_games.jsonl")
    p.add_argument("--grade-depth", type=int, default=12)
    p.add_argument("--deep-depth", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--swing-thresh", type=int, default=150)
    p.add_argument("--before-ceiling", type=int, default=600)
    p.add_argument("--variants", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="training/data/raw/benchmark_blunders_labeled.jsonl")
    args = p.parse_args()
    rng = random.Random(args.seed)

    games = load_games(args.games)
    won = sum(1 for g in games if g["result"] == 1.0)
    drew = sum(1 for g in games if g["result"] == 0.5)
    lost = sum(1 for g in games if g["result"] == 0.0)
    print(f"loaded {len(games)} games ({won}W/{drew}D/{lost}L) from {args.games}", flush=True)

    blunder_pairs = detect_blunders(games, args.grade_depth, args.workers,
                                     args.swing_thresh, skip_won=False,
                                     before_ceiling=args.before_ceiling)
    if not blunder_pairs:
        print("no blunders found; nothing to write.", flush=True)
        return

    blunder_pairs = list(dict.fromkeys(blunder_pairs))
    befores = [b for b, _a in blunder_pairs]
    afters = [a for _b, a in blunder_pairs]
    all_fens = list(befores) + list(afters)
    for fen in befores:
        all_fens.extend(perturb(fen, rng, args.variants))
    all_fens = list(dict.fromkeys(all_fens))
    print(f"{len(blunder_pairs)} blunders -> {len(all_fens)} positions; "
          f"deep-grading at depth {args.deep_depth}...", flush=True)

    labeled = label_positions_parallel(all_fens, DEFAULT_STOCKFISH_PATH,
                                        args.deep_depth, args.workers)

    # depth-consistency diagnostic (same as mine_blunders_v2)
    by_fen_deep = {r["fen"]: (r["cp"] if r["cp"] is not None
                               else (2000 if r["mate"] and r["mate"] > 0 else -2000))
                   for r in labeled}
    we_white_by_fen = {}
    for g in games:
        for fen, we_move in g["plies"]:
            if we_move:
                we_white_by_fen[fen] = g["we_white"]
    confirmed = checked = 0
    for fb, fa in blunder_pairs:
        if fb in by_fen_deep and fa in by_fen_deep and fb in we_white_by_fen:
            checked += 1
            db = our_advantage_cp(by_fen_deep[fb], we_white_by_fen[fb])
            da = our_advantage_cp(by_fen_deep[fa], we_white_by_fen[fb])
            if db - da >= args.swing_thresh / 2:
                confirmed += 1
    if checked:
        print(f"depth-consistency: {confirmed}/{checked} confirmed at depth {args.deep_depth}",
              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in labeled:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(labeled)} labeled positions to {args.out}", flush=True)


if __name__ == "__main__":
    main()
