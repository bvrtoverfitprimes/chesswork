import os
import subprocess
import sys

import chess

CLI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "bestmove_cli.exe")


def get_move(engine, board, time_ms):
    fen = board.fen()
    out = subprocess.run([CLI, engine, fen, str(time_ms)], capture_output=True, text=True, timeout=60)
    return chess.Move.from_uci(out.stdout.strip())


def play_game(white_engine, black_engine, time_ms, max_plies):
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = get_move(engine, board, time_ms)
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    time_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    max_plies = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    a, b = "human", "old"

    score_a = 0.0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(a, b, time_ms, max_plies)
            score_a += r
            print(f"{a}(w) vs {b}(b): {r}", flush=True)
        else:
            r = play_game(b, a, time_ms, max_plies)
            score_a += (1.0 - r)
            print(f"{b}(w) vs {a}(b): {1.0 - r} for {a}", flush=True)

    print(f"\n{a} scored {score_a}/{games} ({100.0 * score_a / games:.1f}%) vs {b} at {time_ms}ms/move")


if __name__ == "__main__":
    main()
