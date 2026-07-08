import os
import subprocess
import sys

import chess
import chess.pgn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "tools", "bestmove_cli.exe")


def get_move(engine, board, time_ms):
    out = subprocess.run([CLI, engine, board.fen(), str(time_ms)], capture_output=True, text=True, timeout=60)
    return chess.Move.from_uci(out.stdout.strip())


def play_game(white_engine, black_engine, time_ms, max_plies):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = white_engine
    game.headers["Black"] = black_engine
    node = game
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = get_move(engine, board, time_ms)
        node = node.add_variation(move)
        board.push(move)
    result = board.result(claim_draw=True)
    game.headers["Result"] = result
    return game, result


def main():
    a, b = "human", "old"
    time_ms = 500
    max_plies = 200
    max_games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    needed_losses = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    losses = []
    for g in range(max_games):
        if g % 2 == 0:
            white, black, a_is_white = a, b, True
        else:
            white, black, a_is_white = b, a, False
        game, result = play_game(white, black, time_ms, max_plies)
        a_won = (result == "1-0" and a_is_white) or (result == "0-1" and not a_is_white)
        a_lost = (result == "0-1" and a_is_white) or (result == "1-0" and not a_is_white)
        outcome = "human WIN" if a_won else ("human LOSS" if a_lost else "draw")
        print(f"game {g+1}: white={white} black={black} result={result} -> {outcome}", flush=True)
        if a_lost:
            losses.append(game)
            print(str(game), flush=True)
            print("", flush=True)
        if len(losses) >= needed_losses:
            break

    print(f"\nfound {len(losses)} human losses out of {g+1} games played")


if __name__ == "__main__":
    main()
