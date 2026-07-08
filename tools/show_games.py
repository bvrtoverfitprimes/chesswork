import argparse
import os
import subprocess

import chess
import chess.engine
import chess.pgn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "tools", "bestmove_cli.exe")
SF_PATH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                        "stockfish-windows-x86-64-avx2.exe")


def our_move(board, time_ms):
    out = subprocess.run([CLI, "human", board.fen(), str(time_ms)], capture_output=True, text=True, timeout=60)
    return chess.Move.from_uci(out.stdout.strip())


def play_vs_old_engine(our_is_white, time_ms, max_plies):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "human_limit" if our_is_white else "old_engine"
    game.headers["Black"] = "old_engine" if our_is_white else "human_limit"
    node = game
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        our_turn = (board.turn == chess.WHITE) == our_is_white
        if our_turn:
            move = our_move(board, time_ms)
        else:
            out = subprocess.run([CLI, "old", board.fen(), str(time_ms)], capture_output=True, text=True, timeout=60)
            move = chess.Move.from_uci(out.stdout.strip())
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result(claim_draw=True)
    return game


def play_vs_stockfish(our_is_white, elo, time_ms, max_plies):
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["White"] = "human_limit" if our_is_white else f"stockfish(elo={elo})"
    game.headers["Black"] = f"stockfish(elo={elo})" if our_is_white else "human_limit"
    node = game
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        our_turn = (board.turn == chess.WHITE) == our_is_white
        if our_turn:
            move = our_move(board, time_ms)
        else:
            result = sf.play(board, chess.engine.Limit(time=time_ms / 1000))
            move = result.move
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result(claim_draw=True)
    sf.quit()
    return game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["old_engine", "stockfish"], required=True)
    parser.add_argument("--elo", type=int, default=1800)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--time-ms", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    for g in range(args.games):
        our_is_white = (g % 2 == 0)
        if args.opponent == "old_engine":
            game = play_vs_old_engine(our_is_white, args.time_ms, args.max_plies)
        else:
            game = play_vs_stockfish(our_is_white, args.elo, args.time_ms, args.max_plies)
        print(f"=== game {g+1} ===")
        print(game)
        print()


if __name__ == "__main__":
    main()
