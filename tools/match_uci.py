import argparse
import os

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR = os.path.join(REPO, "tools", "uci_engine.exe")
SF_PATH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                        "stockfish-windows-x86-64-avx2.exe")


def play_game(our, our_ms, sf, sf_ms, our_is_white, max_plies, game_id):
    board = chess.Board()
    our_limit = chess.engine.Limit(time=our_ms / 1000)
    sf_limit = chess.engine.Limit(time=sf_ms / 1000)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        our_turn = (board.turn == chess.WHITE) == our_is_white
        if our_turn:
            move = our.play(board, our_limit, game=game_id).move
        else:
            move = sf.play(board, sf_limit, game=game_id).move
        if move is None:
            break
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0 if our_is_white else 0.0
    if result == "0-1":
        return 0.0 if our_is_white else 1.0
    return 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--elo", type=int, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--our-time-ms", type=int, default=500)
    parser.add_argument("--sf-time-ms", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=240)
    args = parser.parse_args()

    our = chess.engine.SimpleEngine.popen_uci(OUR)
    if args.threads > 1:
        try:
            our.configure({"Threads": args.threads})
        except Exception:
            pass
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})

    score = 0.0
    for g in range(args.games):
        our_is_white = (g % 2 == 0)
        r = play_game(our, args.our_time_ms, sf, args.sf_time_ms, our_is_white,
                      args.max_plies, object())
        score += r
        side = "w" if our_is_white else "b"
        print(f"game {g+1}: human({side}) vs sf{args.elo}: {r}  running={score}/{g+1}", flush=True)

    our.quit()
    sf.quit()
    pct = 100.0 * score / args.games
    print(f"\nRESULT human {score}/{args.games} ({pct:.1f}%) vs Stockfish UCI_Elo={args.elo} "
          f"@ {args.our_time_ms}ms (threads={args.threads})", flush=True)


if __name__ == "__main__":
    main()
