import argparse
import json
import os
import random
import sys
import time

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.curriculum_positions import endgame_fens

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_STOCKFISH_PATH = os.path.join(
    _REPO_ROOT, "tools", "external", "stockfish", "stockfish", "stockfish-windows-x86-64-avx2.exe")


def stockfish_depth2_game(engine, rng, max_plies, sample_every=3):
    board = chess.Board()
    samples = []
    for ply in range(max_plies):
        if board.is_game_over():
            break
        info = engine.analyse(board, chess.engine.Limit(depth=2))
        pv = info.get("pv")
        if not pv:
            break
        if ply % sample_every == 0 and not board.is_check():
            score = info["score"].white()
            mate = score.mate()
            cp = None if mate is not None else score.score()
            samples.append({"fen": board.fen(), "cp": cp, "mate": mate, "depth": 2})
        board.push(pv[0])
    return samples


def random_opening_fen(rng):
    board = chess.Board()
    plies = rng.randrange(0, 10)
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[rng.randrange(len(legal))])
    return board.fen()


def random_middlegame_fen(rng):
    board = chess.Board()
    plies = 12 + rng.randrange(28)
    for _ in range(plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        board.push(legal[rng.randrange(len(legal))])
    return board.fen()


def random_endgame_fen(rng):
    fens = endgame_fens(rng, 1)
    return fens[0] if fens else random_opening_fen(rng)


def random_phase_fen(rng):
    phase = rng.randrange(3)
    if phase == 0:
        return random_opening_fen(rng)
    if phase == 1:
        return random_middlegame_fen(rng)
    return random_endgame_fen(rng)


def generate_batch(engine, rng, target_positions, sample_every=3, max_plies=60):
    positions = []
    games_played = 0
    t0 = time.time()

    while len(positions) < target_positions:
        if rng.random() < 0.5:
            positions.extend(stockfish_depth2_game(engine, rng, max_plies, sample_every))
            games_played += 1
        else:
            fen = random_phase_fen(rng)
            try:
                info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=2))
            except chess.engine.EngineError:
                continue
            score = info["score"].white()
            mate = score.mate()
            cp = None if mate is not None else score.score()
            positions.append({"fen": fen, "cp": cp, "mate": mate, "depth": 2})

    elapsed = time.time() - t0
    return positions, games_played, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": 1, "Hash": 64})
    rng = random.Random(args.seed)

    positions = []
    games_played = 0
    selfplay_positions = 0
    random_positions = 0
    t0 = time.time()

    while time.time() - t0 < args.seconds:
        if rng.random() < 0.5:
            batch = stockfish_depth2_game(engine, rng, max_plies=60, sample_every=3)
            positions.extend(batch)
            selfplay_positions += len(batch)
            games_played += 1
        else:
            fen = random_phase_fen(rng)
            try:
                info = engine.analyse(chess.Board(fen), chess.engine.Limit(depth=2))
            except chess.engine.EngineError:
                continue
            score = info["score"].white()
            mate = score.mate()
            cp = None if mate is not None else score.score()
            positions.append({"fen": fen, "cp": cp, "mate": mate, "depth": 2})
            random_positions += 1

    elapsed = time.time() - t0
    engine.quit()

    print(f"in {elapsed:.1f}s: {games_played} depth-2 self-play games "
          f"({selfplay_positions} sampled positions), {random_positions} random positions, "
          f"{len(positions)} total labeled positions "
          f"({len(positions) / elapsed:.1f} positions/sec)")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            for p in positions:
                f.write(json.dumps(p) + "\n")
        print(f"wrote {len(positions)} positions to {args.out}")


if __name__ == "__main__":
    main()
