import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.curriculum_positions import build_curriculum

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STOCKFISH_PATH = os.path.join(
    _REPO_ROOT, "tools", "external", "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe")


def _label_worker(fens_chunk, stockfish_path, depth, out_queue):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 64})
    results = []
    for fen in fens_chunk:
        board = chess.Board(fen)
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
        except chess.engine.EngineError:
            continue
        pov_score = info["score"].white()
        mate = pov_score.mate()
        cp = None if mate is not None else pov_score.score()
        results.append({"fen": fen, "cp": cp, "mate": mate, "depth": depth})
    out_queue.put(results)
    try:
        engine.quit()
    except chess.engine.EngineTerminatedError:
        pass


def label_positions_parallel(fens, stockfish_path, depth, num_workers):
    chunks = [fens[i::num_workers] for i in range(num_workers)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = []
    for chunk in chunks:
        if not chunk:
            continue
        p = ctx.Process(target=_label_worker, args=(chunk, stockfish_path, depth, queue))
        p.start()
        procs.append(p)

    all_results = []
    for _ in procs:
        all_results.extend(queue.get())
    for p in procs:
        p.join()
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--depth", type=int, default=14)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selfplay-games", type=int, default=400)
    parser.add_argument("--plies-per-game", type=int, default=60)
    parser.add_argument("--material-positions", type=int, default=6000)
    parser.add_argument("--endgame-positions", type=int, default=6000)
    parser.add_argument("--out", default="training/data/raw/curriculum_labeled.jsonl")
    args = parser.parse_args()

    print("generating curriculum positions...", flush=True)
    fens = build_curriculum(args.seed, args.selfplay_games, args.plies_per_game,
                             args.material_positions, args.endgame_positions)
    print(f"generated {len(fens)} candidate positions", flush=True)

    t0 = time.time()
    results = label_positions_parallel(fens, args.stockfish, args.depth, args.workers)
    elapsed = time.time() - t0
    print(f"labeled {len(results)} positions in {elapsed:.1f}s ({len(results) / max(elapsed, 1e-9):.1f}/s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(results)} labeled positions to {args.out}")


if __name__ == "__main__":
    main()
