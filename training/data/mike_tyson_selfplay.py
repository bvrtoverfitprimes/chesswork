import argparse
import json
import os
import random
import sys
import time

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stockfish_teacher import label_positions_parallel
from build_dataset import white_relative_cp

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SF_PATH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                        "stockfish-windows-x86-64-avx2.exe")

# CPL-based oversampling: a position immediately following a blunder is the highest-signal
# training example (it's exactly what teaches the eval to recognize danger), so we duplicate
# its record in the output in proportion to how bad the move into it was, rather than treating
# every position as equally informative. Capped so one huge swing (e.g. a hung queen) doesn't
# dominate the dataset with near-identical copies of a single position.
CPL_STEP_CP = 150
MAX_OVERSAMPLE = 5


def play_one_game(sf, weak_is_white, weak_depth, teacher_depth, randomize_every, randomize_until_ply,
                   max_plies, rng):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = f"stockfish(depth={weak_depth})" if weak_is_white else f"stockfish(depth={teacher_depth})"
    game.headers["Black"] = f"stockfish(depth={teacher_depth})" if weak_is_white else f"stockfish(depth={weak_depth})"
    node = game

    fens = [board.fen()]
    movers_white = []
    teacher_move_count = 0
    randomized_plies = []

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        weak_turn = (board.turn == chess.WHITE) == weak_is_white

        if ply < 4:
            move = rng.choice(list(board.legal_moves))
        elif weak_turn:
            result = sf.play(board, chess.engine.Limit(depth=weak_depth))
            move = result.move
        else:
            randomize = (teacher_move_count % randomize_every == randomize_every - 1) and ply < randomize_until_ply
            if randomize:
                move = rng.choice(list(board.legal_moves))
                randomized_plies.append(ply)
            else:
                result = sf.play(board, chess.engine.Limit(depth=teacher_depth))
                move = result.move
            teacher_move_count += 1

        movers_white.append(board.turn == chess.WHITE)
        node = node.add_variation(move)
        board.push(move)
        fens.append(board.fen())

    game.headers["Result"] = board.result(claim_draw=True)
    game.headers["RandomizedTeacherPlies"] = ",".join(str(p) for p in randomized_plies)
    return game, fens, movers_white


def oversample_weight(cpl):
    return 1 + min(MAX_OVERSAMPLE - 1, int(max(0, cpl) // CPL_STEP_CP))


def label_game_with_weights(fens, movers_white, sf_path, depth, workers):
    graded = label_positions_parallel(fens, sf_path, depth, workers)
    cp_by_fen = {r["fen"]: white_relative_cp(r) for r in graded}
    row_by_fen = {r["fen"]: r for r in graded}

    weighted_rows = []
    for i, fen in enumerate(fens):
        row = row_by_fen.get(fen)
        if row is None:
            continue
        if i == 0:
            weight = 1
        else:
            prev_cp = cp_by_fen.get(fens[i - 1])
            cur_cp = cp_by_fen.get(fen)
            if prev_cp is None or cur_cp is None:
                weight = 1
            else:
                white_moved = movers_white[i - 1]
                cpl = (prev_cp - cur_cp) if white_moved else (cur_cp - prev_cp)
                weight = oversample_weight(cpl)
        for _ in range(weight):
            weighted_rows.append(row)
    return weighted_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--weak-depth", type=int, default=2)
    parser.add_argument("--teacher-depth", type=int, default=10)
    parser.add_argument("--randomize-every", type=int, default=4)
    parser.add_argument("--randomize-until-ply", type=int, default=75)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grade-workers", type=int, default=6)
    parser.add_argument("--out", default="training/data/raw/mike_tyson.jsonl")
    parser.add_argument("--pgn-out", default="training/data/raw/mike_tyson_games.pgn")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"Threads": 1})

    os.makedirs(os.path.dirname(args.pgn_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_labeled = 0
    t0 = time.time()
    gameplay_time = 0.0
    grade_time = 0.0
    try:
        with open(args.pgn_out, "w") as pgn_f, open(args.out, "w") as out_f:
            for g in range(args.games):
                weak_is_white = rng.random() < 0.5
                g0 = time.time()
                game, fens, movers_white = play_one_game(sf, weak_is_white, args.weak_depth, args.teacher_depth,
                                                          args.randomize_every, args.randomize_until_ply,
                                                          args.max_plies, rng)
                g_elapsed = time.time() - g0
                gameplay_time += g_elapsed

                pgn_f.write(str(game) + "\n\n")
                pgn_f.flush()

                gr0 = time.time()
                weighted_rows = label_game_with_weights(fens, movers_white, SF_PATH, args.teacher_depth,
                                                         args.grade_workers)
                gr_elapsed = time.time() - gr0
                grade_time += gr_elapsed
                for r in weighted_rows:
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()
                all_labeled += len(weighted_rows)

                print(f"game {g + 1}/{args.games}: result={game.headers['Result']} plies={len(fens) - 1} "
                      f"weak_color={'white' if weak_is_white else 'black'} play_time={g_elapsed:.1f}s "
                      f"grade_time={gr_elapsed:.1f}s rows={len(weighted_rows)} "
                      f"total_elapsed={time.time() - t0:.1f}s", flush=True)
    finally:
        sf.quit()

    total_elapsed = time.time() - t0
    print(f"TOTAL: {total_elapsed:.1f}s for {args.games} games "
          f"({gameplay_time:.1f}s gameplay + {grade_time:.1f}s grading), {all_labeled} labeled rows (with oversampling)")
    print(f"wrote games to {args.pgn_out}, labeled data to {args.out}")


if __name__ == "__main__":
    main()
