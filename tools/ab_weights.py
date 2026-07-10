import argparse
import os

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR = os.path.join(REPO, "tools", "uci_engine.exe")


def open_engine(weights, threads=1):
    os.environ["LIMIT_WEIGHTS"] = weights
    eng = chess.engine.SimpleEngine.popen_uci(OUR)
    if threads > 1:
        try:
            eng.configure({"Threads": threads})
        except Exception:
            pass
    return eng


def play_game(a, b, a_ms, b_ms, a_is_white, max_plies, gid):
    board = chess.Board()
    la = chess.engine.Limit(time=a_ms / 1000)
    lb = chess.engine.Limit(time=b_ms / 1000)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        a_turn = (board.turn == chess.WHITE) == a_is_white
        eng, lim = (a, la) if a_turn else (b, lb)
        mv = eng.play(board, lim, game=gid).move
        if mv is None:
            break
        board.push(mv)
    res = board.result(claim_draw=True)
    if res == "1-0":
        return 1.0 if a_is_white else 0.0
    if res == "0-1":
        return 0.0 if a_is_white else 1.0
    return 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="weights file for engine A (new)")
    p.add_argument("--b", required=True, help="weights file for engine B (baseline)")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--ms", type=int, default=400)
    p.add_argument("--a-threads", type=int, default=1)
    p.add_argument("--b-threads", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=240)
    args = p.parse_args()

    a = open_engine(args.a, args.a_threads)
    b = open_engine(args.b, args.b_threads)
    score = 0.0
    for g in range(args.games):
        a_white = (g % 2 == 0)
        r = play_game(a, b, args.ms, args.ms, a_white, args.max_plies, object())
        score += r
        print(f"game {g+1}: A({'w' if a_white else 'b'}) {r}  running A={score}/{g+1}", flush=True)
    a.quit()
    b.quit()
    pct = 100.0 * score / args.games
    print(f"\nRESULT A(new)={score}/{args.games} ({pct:.1f}%) vs B(baseline)  @ {args.ms}ms", flush=True)


if __name__ == "__main__":
    main()
