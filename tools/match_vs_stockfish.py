import argparse
import os
import subprocess

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "tools", "bestmove_cli.exe")
SF_PATH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                        "stockfish-windows-x86-64-avx2.exe")


def our_move(engine_label, board, time_ms):
    out = subprocess.run([CLI, engine_label, board.fen(), str(time_ms)],
                          capture_output=True, text=True, timeout=60)
    return chess.Move.from_uci(out.stdout.strip())


def play_game(our_label, our_time_ms, sf_engine, sf_time_ms, our_is_white, max_plies):
    board = chess.Board()
    for _ in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        our_turn = (board.turn == chess.WHITE) == our_is_white
        if our_turn:
            move = our_move(our_label, board, our_time_ms)
        else:
            result = sf_engine.play(board, chess.engine.Limit(time=sf_time_ms / 1000))
            move = result.move
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if our_is_white else 0.0
    if result == "0-1":
        return 0.0 if our_is_white else 1.0
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="human")
    parser.add_argument("--elo", type=int, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--our-time-ms", type=int, default=500)
    parser.add_argument("--sf-time-ms", type=int, default=500)
    parser.add_argument("--max-plies", type=int, default=200)
    args = parser.parse_args()

    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})

    score = 0.0
    for g in range(args.games):
        our_is_white = (g % 2 == 0)
        r = play_game(args.engine, args.our_time_ms, sf, args.sf_time_ms, our_is_white, args.max_plies)
        score += r
        side = "w" if our_is_white else "b"
        print(f"game {g+1}: {args.engine}({side}) vs stockfish(elo={args.elo}): {r}", flush=True)

    sf.quit()
    print(f"\n{args.engine} scored {score}/{args.games} ({100.0*score/args.games:.1f}%) "
          f"vs Stockfish (UCI_Elo={args.elo}) at {args.our_time_ms}ms/{args.sf_time_ms}ms per move")


if __name__ == "__main__":
    main()
