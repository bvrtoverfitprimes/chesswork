"""Rigorous, unattended engine-strength benchmark, following practice closer to
serious engine testing (cutechess-cli style) than a quick sweep:

- Every game starts from the TRUE starting position -- no opening book, no
  hardcoded/injected positions of any kind (the engine must be fully
  self-reliant; see project rule). Variance is instead controlled by sample
  size and by reporting Elo with a confidence interval, not a raw win rate.
- Multiple opponents: Stockfish at several UCI_LimitStrength Elo targets, PLUS
  a fixed-node (not time-limited, not strength-limited) Stockfish reference,
  which sidesteps the timing-instability found in UCI_LimitStrength at very
  short time controls.
- Identical Threads=1 and a fixed Hash setting on the Stockfish side for every
  opponent (our engine's Hash UCI option is currently advertised but not
  actually wired to anything -- its transposition table size is fixed at
  compile time; this asymmetry is real and reported, not hidden).
- Unattended-safe: small batches, append-after-every-game logging, a running
  summary json, and a full per-game move/FEN trail saved for every game so the
  games can later be fed into the blunder-mining pipeline as training
  positions (mine_blunders.py's detect_blunders() expects exactly this
  {plies: [(fen, we_move)...], we_white, result} shape).
"""
import argparse
import json
import os
import sys
import time
import traceback

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.elo_estimate import estimate, format_estimate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUR = os.path.join(REPO, "tools", "uci_engine.exe")
STOCKFISH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                          "stockfish-windows-x86-64-avx2.exe")


def open_ours(hash_mb, weights=None):
    env = dict(os.environ)
    if weights:
        env["LIMIT_WEIGHTS"] = os.path.abspath(weights)
    eng = chess.engine.SimpleEngine.popen_uci(OUR, env=env)
    try:
        eng.configure({"Threads": 1, "Hash": hash_mb})
    except Exception:
        pass  # Hash option is advertised but not wired on our side; harmless no-op.
    return eng


def open_stockfish_elo(elo, hash_mb):
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    eng.configure({"UCI_LimitStrength": True, "UCI_Elo": elo, "Threads": 1, "Hash": hash_mb})
    return eng


def open_stockfish_nodes(hash_mb):
    """Unrestricted Stockfish (no UCI_LimitStrength) -- strength is whatever a
    fixed node budget per move gives it. Deterministic-ish and immune to the
    UCI_LimitStrength short-time-control instability found earlier."""
    eng = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    eng.configure({"UCI_LimitStrength": False, "Threads": 1, "Hash": hash_mb})
    return eng


def play_game(our, opp, our_limit, opp_limit, our_is_white, max_plies):
    board = chess.Board()
    plies = []  # (fen_before, we_move) -- same shape mine_blunders.py expects
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        we_move = (board.turn == chess.WHITE) == our_is_white
        plies.append((board.fen(), we_move))
        eng, lim = (our, our_limit) if we_move else (opp, opp_limit)
        mv = eng.play(board, lim, game=object()).move
        if mv is None:
            break
        board.push(mv)
    result = board.result(claim_draw=True)
    if result == "1-0":
        our_pov = 1.0 if our_is_white else 0.0
    elif result == "0-1":
        our_pov = 0.0 if our_is_white else 1.0
    else:
        our_pov = 0.5
    return {"plies": plies, "we_white": our_is_white, "result": our_pov,
            "final_fen": board.fen(), "ply_count": board.ply()}


def run_opponent(label, open_opp_fn, our_limit, opp_limit, n_games, max_plies,
                  hash_mb, weights, games_out_path, log):
    our = open_ours(hash_mb, weights)
    opp = open_opp_fn()
    wins = draws = losses = 0
    for g in range(n_games):
        our_white = (g % 2 == 0)
        try:
            game = play_game(our, opp, our_limit, opp_limit, our_white, max_plies)
        except Exception as e:
            log(f"[{label}] game {g+1}/{n_games}: ERROR ({e}) -- restarting engines")
            try:
                our.quit()
            except Exception:
                pass
            try:
                opp.quit()
            except Exception:
                pass
            our = open_ours(hash_mb, weights)
            opp = open_opp_fn()
            continue

        if game["result"] == 1.0:
            wins += 1
        elif game["result"] == 0.5:
            draws += 1
        else:
            losses += 1

        with open(games_out_path, "a") as f:
            f.write(json.dumps({"opponent": label, **game}) + "\n")

        log(f"[{label}] game {g+1}/{n_games}: result={game['result']} "
            f"running={wins}W/{draws}D/{losses}L")

    our.quit()
    opp.quit()
    return wins, draws, losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games-per-opponent", type=int, default=40)
    p.add_argument("--our-ms", type=int, default=1000)
    p.add_argument("--sf-ms", type=int, default=1000)
    p.add_argument("--sf-nodes", type=int, default=2_000_000,
                    help="node budget/move for the unrestricted (non-elo-limited) Stockfish reference")
    p.add_argument("--elo-levels", type=int, nargs="+", default=[2200, 2300, 2400])
    p.add_argument("--include-unrestricted", action="store_true", default=True)
    p.add_argument("--hash-mb", type=int, default=128)
    p.add_argument("--max-plies", type=int, default=240)
    p.add_argument("--weights", default=None)
    p.add_argument("--games-out", default="training/data/raw/rigorous_benchmark_games.jsonl")
    p.add_argument("--log", default="tools/elo_logs/rigorous_benchmark_progress.log")
    p.add_argument("--summary", default="tools/elo_logs/rigorous_benchmark_summary.json")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.games_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(args.log, "a") as f:
            f.write(line + "\n")

    our_limit = chess.engine.Limit(time=args.our_ms / 1000)
    opp_time_limit = chess.engine.Limit(time=args.sf_ms / 1000)
    opp_node_limit = chess.engine.Limit(nodes=args.sf_nodes)

    opponents = [(f"sf_elo_{elo}", lambda elo=elo: open_stockfish_elo(elo, args.hash_mb), opp_time_limit)
                 for elo in args.elo_levels]
    if args.include_unrestricted:
        opponents.append(("sf_unrestricted_nodes",
                           lambda: open_stockfish_nodes(args.hash_mb), opp_node_limit))

    results = {}
    start = time.time()
    log(f"starting rigorous benchmark: {len(opponents)} opponents x "
        f"{args.games_per_opponent} games, our={args.our_ms}ms, opponents={[o[0] for o in opponents]}")

    for label, open_fn, opp_limit in opponents:
        log(f"=== opponent {label}: {args.games_per_opponent} games ===")
        try:
            w, d, l = run_opponent(label, open_fn, our_limit, opp_limit,
                                    args.games_per_opponent, args.max_plies,
                                    args.hash_mb, args.weights, args.games_out, log)
        except Exception:
            log(f"opponent {label} FAILED: {traceback.format_exc()}")
            continue
        est = estimate(w, d, l)
        results[label] = est
        log(format_estimate(label, est))
        with open(args.summary, "w") as f:
            json.dump({
                "elapsed_hours": round((time.time() - start) / 3600, 2),
                "results": results,
            }, f, indent=2)

    log("=== FINAL RESULTS ===")
    for label, est in results.items():
        log(format_estimate(label, est))
    log(f"all games appended to {args.games_out} (usable as blunder-mining input games)")


if __name__ == "__main__":
    main()
