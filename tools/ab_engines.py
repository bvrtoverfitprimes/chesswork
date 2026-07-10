"""A/B two engine binaries (each optionally with its own weights file) against
each other, or one binary against Stockfish at a calibrated Elo.

Quick-filter protocol (per user instruction): run a small batch first (default
4 games); only continue to the full sample (default +4 more = 8 total) if the
quick batch isn't a clear loss. This avoids burning a full 8-game sample on a
candidate that's obviously worse.
"""
import argparse
import os

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCKFISH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                          "stockfish-windows-x86-64-avx2.exe")


def open_ours(binary, weights, threads=1):
    env = dict(os.environ)
    if weights:
        env["LIMIT_WEIGHTS"] = os.path.abspath(weights)
    binary = os.path.abspath(binary)
    eng = chess.engine.SimpleEngine.popen_uci(binary, timeout=60, env=env)
    if threads > 1:
        try:
            eng.configure({"Threads": threads})
        except Exception:
            pass
    return eng


def open_stockfish(elo, timeout):
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH, timeout=timeout)
    eng.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1})
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


def run_batch(a, b, ms, max_plies, n, start_game_no, label):
    score = 0.0
    for i in range(n):
        g = start_game_no + i
        a_white = (g % 2 == 0)
        r = play_game(a, b, ms, ms, a_white, max_plies, object())
        score += r
        print(f"[{label}] game {g + 1}: A({'w' if a_white else 'b'}) {r}  "
              f"running A={score}/{i + 1}", flush=True)
    return score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-binary", required=True, help="engine A binary (candidate)")
    p.add_argument("--a-weights", default=None)
    p.add_argument("--a-threads", type=int, default=1)
    p.add_argument("--b-binary", default=None, help="engine B binary (baseline); omit to play Stockfish instead")
    p.add_argument("--b-weights", default=None)
    p.add_argument("--b-threads", type=int, default=1)
    p.add_argument("--sf-elo", type=int, default=None, help="play Stockfish at this Elo instead of engine B")
    p.add_argument("--ms", type=int, default=500)
    p.add_argument("--max-plies", type=int, default=240)
    p.add_argument("--quick-games", type=int, default=4)
    p.add_argument("--full-games", type=int, default=8)
    p.add_argument("--quick-min-score", type=float, default=1.0,
                    help="if A's score in the quick batch is < this, stop early (clear loss)")
    args = p.parse_args()

    a = open_ours(args.a_binary, args.a_weights, args.a_threads)
    if args.sf_elo is not None:
        b = open_stockfish(args.sf_elo, timeout=max(30.0, 4 * args.ms / 1000))
        label_b = f"sf{args.sf_elo}"
    else:
        b = open_ours(args.b_binary, args.b_weights, args.b_threads)
        label_b = "B"

    quick_n = min(args.quick_games, args.full_games)
    score = run_batch(a, b, args.ms, args.max_plies, quick_n, 0, "quick")
    played = quick_n

    if score < args.quick_min_score:
        print(f"\nSTOPPING EARLY: quick batch score {score}/{quick_n} vs {label_b} "
              f"is below threshold {args.quick_min_score} -- candidate looks clearly worse.",
              flush=True)
    else:
        remaining = args.full_games - quick_n
        if remaining > 0:
            score += run_batch(a, b, args.ms, args.max_plies, remaining, quick_n, "extend")
            played += remaining

    a.quit()
    b.quit()
    pct = 100.0 * score / played if played else 0.0
    print(f"\nRESULT A={score}/{played} ({pct:.1f}%) vs {label_b} @ {args.ms}ms "
          f"(a_threads={args.a_threads}, b_threads={args.b_threads})", flush=True)


if __name__ == "__main__":
    main()
