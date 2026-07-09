"""Unattended, long-running blunder collection: play small batches of games vs
Stockfish (alternating Elo levels), grade every position, detect where OUR move
dropped our own eval (a blunder - including in games we won/drew), perturb each
blunder into near-neighbours, deep-grade everything, and append to an
ever-growing weak-point jsonl. Designed to run for hours unattended:

- Small batches (fresh engines each cycle) so one stuck/crashed engine only
  costs a few games, not the whole run.
- Every write is append + flush immediately after each cycle, so a crash or
  forced stop at any point loses at most the in-progress cycle.
- A running summary json is rewritten each cycle for an at-a-glance status.
- Any exception in a cycle is logged and the loop continues (never dies early).
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mine_blunders import play_games, detect_blunders, perturb
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--elos", type=int, nargs="+", default=[2100, 1800])
    p.add_argument("--games-per-cycle", type=int, default=4)
    p.add_argument("--our-ms", type=int, default=300)
    p.add_argument("--sf-ms", type=int, default=300)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--grade-depth", type=int, default=12)
    p.add_argument("--deep-depth", type=int, default=14)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--swing-thresh", type=int, default=150)
    p.add_argument("--before-ceiling", type=int, default=600)
    p.add_argument("--variants", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="training/data/raw/overnight_blunders.jsonl")
    p.add_argument("--raw-games-out", default="training/data/raw/overnight_games.jsonl")
    p.add_argument("--log", default="training/data/raw/overnight_progress.log")
    p.add_argument("--summary", default="training/data/raw/overnight_summary.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = random.Random(args.seed)
    start = time.time()
    deadline = start + args.hours * 3600

    total_games = 0
    total_blunders = 0
    total_positions = 0
    total_won = total_drew = total_lost = 0
    cycle = 0
    elo_idx = 0

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(args.log, "a") as f:
            f.write(line + "\n")

    def write_summary():
        with open(args.summary, "w") as f:
            json.dump({
                "cycle": cycle,
                "elapsed_hours": round((time.time() - start) / 3600, 2),
                "remaining_hours": round((deadline - time.time()) / 3600, 2),
                "total_games": total_games,
                "record": {"won": total_won, "drew": total_drew, "lost": total_lost},
                "total_blunders": total_blunders,
                "total_weak_point_positions": total_positions,
                "out_file": args.out,
            }, f, indent=2)

    log(f"starting overnight collection: {args.hours}h budget, elos={args.elos}, "
        f"{args.games_per_cycle} games/cycle, our={args.our_ms}ms sf={args.sf_ms}ms "
        f"threads={args.threads}, out={args.out}")
    write_summary()

    while time.time() < deadline:
        cycle += 1
        elo = args.elos[elo_idx % len(args.elos)]
        elo_idx += 1
        try:
            log(f"cycle {cycle}: playing {args.games_per_cycle} games vs Stockfish@{elo}...")
            games = play_games(args.games_per_cycle, args.our_ms, args.sf_ms, elo,
                                args.threads, args.max_plies)

            with open(args.raw_games_out, "a") as f:
                for g in games:
                    f.write(json.dumps({"elo": elo, **g}) + "\n")

            won = sum(1 for g in games if g["result"] == 1.0)
            drew = sum(1 for g in games if g["result"] == 0.5)
            lost = sum(1 for g in games if g["result"] == 0.0)
            total_games += len(games)
            total_won += won
            total_drew += drew
            total_lost += lost
            log(f"cycle {cycle}: {won}W/{drew}D/{lost}L vs {elo} "
                f"(total games so far: {total_games})")

            blunders = detect_blunders(games, args.grade_depth, args.workers,
                                        args.swing_thresh, skip_won=False,
                                        before_ceiling=args.before_ceiling)
            if not blunders:
                log(f"cycle {cycle}: no blunders found this cycle")
                write_summary()
                continue

            blunders = list(dict.fromkeys(blunders))
            all_fens = list(blunders)
            for fen in blunders:
                all_fens.extend(perturb(fen, rng, args.variants))
            all_fens = list(dict.fromkeys(all_fens))

            log(f"cycle {cycle}: {len(blunders)} blunders -> {len(all_fens)} positions, "
                f"deep-grading at depth {args.deep_depth}...")
            labeled = label_positions_parallel(all_fens, DEFAULT_STOCKFISH_PATH,
                                                args.deep_depth, args.workers)

            with open(args.out, "a") as f:
                for r in labeled:
                    f.write(json.dumps(r) + "\n")

            total_blunders += len(blunders)
            total_positions += len(labeled)
            elapsed_h = (time.time() - start) / 3600
            remaining_h = (deadline - time.time()) / 3600
            log(f"cycle {cycle}: wrote {len(labeled)} positions "
                f"(total positions: {total_positions}, total blunders: {total_blunders}) "
                f"| elapsed={elapsed_h:.2f}h remaining={remaining_h:.2f}h")
            write_summary()

        except Exception as e:
            log(f"cycle {cycle}: ERROR: {e}\n{traceback.format_exc()}")
            write_summary()
            time.sleep(5)
            continue

    log(f"DONE. total_games={total_games} ({total_won}W/{total_drew}D/{total_lost}L) "
        f"total_blunders={total_blunders} total_positions={total_positions}")
    log(f"output: {args.out}")
    write_summary()


if __name__ == "__main__":
    main()
