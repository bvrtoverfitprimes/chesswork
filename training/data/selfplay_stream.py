import argparse
import json
import os
import random
import sys
import time

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import DEFAULT_STOCKFISH_PATH

MAX_PLIES_PER_GAME = 300  # safety net only (e.g. long shuffling draws) — games are meant to
                          # actually finish now, not get cut off mid-play

# Per-ply schedule:
#   plies 0-3   : forced random move (pure opening variance)
#   plies 4-7   : forced best move at depth 7 (settles the position after the random opening)
#   plies 8-49  : depth 2, mostly best moves with a random move every RANDOM_PERIOD plies
#   ply 50+     : forced best move at depth 7 (was 10 — cheaper, still deeper than the depth-2
#                 middlegame phase, targets the project's known endgame-conversion weakness)
OPENING_RANDOM_PLIES = 4
OPENING_SETTLE_PLIES = 4     # plies OPENING_RANDOM_PLIES .. OPENING_RANDOM_PLIES+SETTLE-1
OPENING_SETTLE_DEPTH = 7
RANDOM_UNTIL_PLY = 50
RANDOM_PERIOD = 9             # every RANDOM_PERIOD plies is random (odd period — see note below)
ENDGAME_DEPTH = 7
GRADE_DEPTH = 4                # separate grading pass over every generated position, independent
                                # of whatever depth was used to pick the move during generation


def ply_plan(ply, rng, legal, info_pv):
    """Returns (search_depth_for_move_selection, is_random_move) for this ply — controls game
    generation only. Labeling/grading is a separate pass, see GRADE_DEPTH."""
    if ply < OPENING_RANDOM_PLIES:
        return 2, True
    if ply < OPENING_RANDOM_PLIES + OPENING_SETTLE_PLIES:
        return OPENING_SETTLE_DEPTH, False
    if ply < RANDOM_UNTIL_PLY:
        # RANDOM_PERIOD is odd on purpose: since ply parity determines side to move (even=White,
        # odd=Black), an odd period means each successive random move lands on the opposite
        # color from the last, so both sides get random moves roughly equally over a game
        # rather than one side always drawing them (an even period would hit the same color
        # every time).
        return 2, (ply % RANDOM_PERIOD == RANDOM_PERIOD - 1)
    return ENDGAME_DEPTH, False


def run_worker(worker_id, stockfish_path, minutes, out_path, seed):
    rng = random.Random(seed)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 32})

    deadline = time.time() + minutes * 60
    games = 0
    positions = 0

    with open(out_path, "a") as out_f:
        while time.time() < deadline:
            board = chess.Board()
            ply = 0
            while not board.is_game_over(claim_draw=True) and ply < MAX_PLIES_PER_GAME and time.time() < deadline:
                legal = list(board.legal_moves)
                if not legal:
                    break
                search_depth, is_random = ply_plan(ply, rng, legal, None)
                try:
                    info = engine.analyse(board, chess.engine.Limit(depth=search_depth))
                except chess.engine.EngineError:
                    break

                # Grading is a separate pass at a fixed depth, independent of whichever depth
                # was used above to pick the move — this is the label actually written out.
                grade_info = engine.analyse(board, chess.engine.Limit(depth=GRADE_DEPTH))
                pov_score = grade_info["score"].white()
                mate = pov_score.mate()
                cp = None if mate is not None else pov_score.score()
                out_f.write(json.dumps({"fen": board.fen(), "cp": cp, "mate": mate,
                                         "depth": GRADE_DEPTH}) + "\n")
                positions += 1

                if is_random:
                    move = rng.choice(legal)
                else:
                    pv = info.get("pv")
                    move = pv[0] if pv else rng.choice(legal)
                board.push(move)
                ply += 1
            games += 1
            if games % 20 == 0:
                out_f.flush()

    engine.quit()
    print(f"[worker {worker_id}] done: {games} games, {positions} positions -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default=DEFAULT_STOCKFISH_PATH)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_worker(args.worker_id, args.stockfish, args.minutes, args.out, args.seed + args.worker_id)


if __name__ == "__main__":
    main()
