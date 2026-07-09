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

    # Default python-chess protocol timeout (10s) is tighter than our_time_ms can
    # need once SMP thread spawn/join overhead is added under system load; give
    # it real headroom rather than crashing the whole sweep on one slow move.
    engine_timeout = max(30.0, 4 * args.our_time_ms / 1000)
    our = chess.engine.SimpleEngine.popen_uci(OUR, timeout=engine_timeout)
    if args.threads > 1:
        try:
            our.configure({"Threads": args.threads})
        except Exception:
            pass
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH, timeout=engine_timeout)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})

    score = 0.0
    played = 0
    for g in range(args.games):
        our_is_white = (g % 2 == 0)
        try:
            r = play_game(our, args.our_time_ms, sf, args.sf_time_ms, our_is_white,
                          args.max_plies, object())
        except Exception as e:
            print(f"game {g+1}: ERROR ({e}) - skipping, restarting engines", flush=True)
            try:
                our.quit()
            except Exception:
                pass
            try:
                sf.quit()
            except Exception:
                pass
            our = chess.engine.SimpleEngine.popen_uci(OUR, timeout=engine_timeout)
            if args.threads > 1:
                try:
                    our.configure({"Threads": args.threads})
                except Exception:
                    pass
            sf = chess.engine.SimpleEngine.popen_uci(SF_PATH, timeout=engine_timeout)
            sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo, "Threads": 1})
            continue
        score += r
        played += 1
        side = "w" if our_is_white else "b"
        print(f"game {g+1}: human({side}) vs sf{args.elo}: {r}  running={score}/{played}", flush=True)

    our.quit()
    sf.quit()
    pct = 100.0 * score / played if played else 0.0
    print(f"\nRESULT human {score}/{played} ({pct:.1f}%) vs Stockfish UCI_Elo={args.elo} "
          f"@ {args.our_time_ms}ms (threads={args.threads})", flush=True)


if __name__ == "__main__":
    main()
