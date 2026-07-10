"""Place the classical champion's true level with a FIXED-NODE Stockfish bracket
(calibration-free, unlike UCI_LimitStrength). Our engine at fixed movetime vs
Stockfish at N nodes/move, several N, to find where we cross ~50%. Balanced colors."""
import argparse
import math
import os

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "tools", "uci_engine_raw.exe")
SF = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                   "stockfish-windows-x86-64-avx2.exe")


def play(our, sf, our_ms, nodes, a_white, max_plies=240):
    board = chess.Board()
    ol = chess.engine.Limit(time=our_ms / 1000)
    sl = chess.engine.Limit(nodes=nodes)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        our_turn = (board.turn == chess.WHITE) == a_white
        mv = (our.play(board, ol, game=object()).move if our_turn
              else sf.play(board, sl, game=object()).move)
        if mv is None:
            break
        board.push(mv)
    r = board.result(claim_draw=True)
    if r == "1-0":
        return 1.0 if a_white else 0.0
    if r == "0-1":
        return 0.0 if a_white else 1.0
    return 0.5


def elo(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -400 * math.log10(1 / p - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, nargs="+", default=[1000, 2500, 6000, 15000, 40000])
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--our-ms", type=int, default=1000)
    args = ap.parse_args()

    env = dict(os.environ); env["RAW_WEIGHT"] = "1"
    our = chess.engine.SimpleEngine.popen_uci(RAW, env=env)
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"Threads": 1, "Hash": 128})

    for nodes in args.nodes:
        score = w = d = l = 0
        for g in range(args.games):
            r = play(our, sf, args.our_ms, nodes, g % 2 == 0)
            score += r
            w += r == 1.0; d += r == 0.5; l += r == 0.0
        pct = 100.0 * score / args.games
        e = elo(score / args.games)
        print(f"SF {nodes:6d} nodes/move: {pct:5.1f}% ({w:.0f}W/{d:.0f}D/{l:.0f}L)  "
              f"our Elo vs this = {e:+.0f}", flush=True)
    our.quit()
    sf.quit()


if __name__ == "__main__":
    main()
