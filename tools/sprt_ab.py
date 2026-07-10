"""SPRT A/B test between two engine binaries/weights, with parallel games.

Sequential Probability Ratio Test (standard engine-testing practice): instead
of a fixed N, stop as soon as the accumulated evidence crosses a likelihood
bound. H0: elo diff <= elo0, H1: elo diff >= elo1. Uses the trinomial
(win/draw/loss) GSPRT approximation used by fishtest/cutechess.

Games run in parallel worker pairs (each worker owns one A and one B engine
process), alternating colors per game index. All games start from the true
start position (project rule: no opening books, no hardcoded positions).
"""
import argparse
import math
import os
import threading

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gsprt_llr(wins, draws, losses, elo0, elo1):
    """Log-likelihood ratio via the GSPRT normal approximation (fishtest-style)."""
    n = wins + draws + losses
    if n == 0 or wins == n or losses == n:
        return 0.0
    w, d, l = wins / n, draws / n, losses / n
    score = w + 0.5 * d
    var = w * (1 - score) ** 2 + d * (0.5 - score) ** 2 + l * (0 - score) ** 2
    if var <= 0:
        return 0.0
    def elo_to_score(e):
        return 1.0 / (1.0 + 10 ** (-e / 400.0))
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return (s1 - s0) * (2 * score - s0 - s1) * n / (2 * var)


def open_engine(binary, weights, threads):
    env = dict(os.environ)
    if weights:
        env["LIMIT_WEIGHTS"] = os.path.abspath(weights)
    eng = chess.engine.SimpleEngine.popen_uci(os.path.abspath(binary), timeout=60, env=env)
    if threads > 1:
        try:
            eng.configure({"Threads": threads})
        except Exception:
            pass
    return eng


def play_game(a, b, ms, a_is_white, max_plies):
    board = chess.Board()
    lim = chess.engine.Limit(time=ms / 1000)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        a_turn = (board.turn == chess.WHITE) == a_is_white
        mv = (a if a_turn else b).play(board, lim, game=object()).move
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
    p.add_argument("--a-binary", required=True)
    p.add_argument("--a-weights", default=None)
    p.add_argument("--b-binary", required=True)
    p.add_argument("--b-weights", default=None)
    p.add_argument("--threads", type=int, default=1, help="engine threads per side")
    p.add_argument("--ms", type=int, default=500)
    p.add_argument("--max-plies", type=int, default=240)
    p.add_argument("--concurrency", type=int, default=2, help="parallel game workers")
    p.add_argument("--elo0", type=float, default=0.0)
    p.add_argument("--elo1", type=float, default=50.0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--max-games", type=int, default=100)
    args = p.parse_args()

    lower = math.log(args.beta / (1 - args.alpha))
    upper = math.log((1 - args.beta) / args.alpha)

    lock = threading.Lock()
    state = {"w": 0, "d": 0, "l": 0, "game_no": 0, "stop": False}

    def worker():
        a = open_engine(args.a_binary, args.a_weights, args.threads)
        b = open_engine(args.b_binary, args.b_weights, args.threads)
        try:
            while True:
                with lock:
                    if state["stop"] or state["game_no"] >= args.max_games:
                        return
                    g = state["game_no"]
                    state["game_no"] += 1
                try:
                    r = play_game(a, b, args.ms, g % 2 == 0, args.max_plies)
                except Exception as e:
                    with lock:
                        print(f"game {g + 1}: ERROR {e}, restarting engines", flush=True)
                    for eng in (a, b):
                        try:
                            eng.quit()
                        except Exception:
                            pass
                    a = open_engine(args.a_binary, args.a_weights, args.threads)
                    b = open_engine(args.b_binary, args.b_weights, args.threads)
                    continue
                with lock:
                    if r == 1.0:
                        state["w"] += 1
                    elif r == 0.5:
                        state["d"] += 1
                    else:
                        state["l"] += 1
                    w, d, l = state["w"], state["d"], state["l"]
                    llr = gsprt_llr(w, d, l, args.elo0, args.elo1)
                    n = w + d + l
                    print(f"game {g + 1}: A {'w' if g % 2 == 0 else 'b'} {r} | "
                          f"{w}W/{d}D/{l}L ({100 * (w + 0.5 * d) / n:.1f}%) "
                          f"LLR={llr:.2f} [{lower:.2f},{upper:.2f}]", flush=True)
                    if llr >= upper:
                        state["stop"] = True
                        print(f"\nSPRT: H1 ACCEPTED (A is at least ~{args.elo1:+.0f} Elo) "
                              f"after {n} games", flush=True)
                    elif llr <= lower:
                        state["stop"] = True
                        print(f"\nSPRT: H0 ACCEPTED (A is not better than ~{args.elo0:+.0f} Elo) "
                              f"after {n} games", flush=True)
        finally:
            for eng in (a, b):
                try:
                    eng.quit()
                except Exception:
                    pass

    ts = [threading.Thread(target=worker) for _ in range(args.concurrency)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()

    w, d, l = state["w"], state["d"], state["l"]
    n = max(w + d + l, 1)
    llr = gsprt_llr(w, d, l, args.elo0, args.elo1)
    print(f"\nFINAL: {w}W/{d}D/{l}L ({100 * (w + 0.5 * d) / n:.1f}%) LLR={llr:.2f} "
          f"(bounds [{lower:.2f},{upper:.2f}]) after {n} games", flush=True)
    if lower < llr < upper:
        print("SPRT: inconclusive at max-games (treat as no clear win)", flush=True)


if __name__ == "__main__":
    main()
