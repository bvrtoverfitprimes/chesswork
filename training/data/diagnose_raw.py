"""Stockfish-guided diagnosis of raw_engine blunders, per the user's method:
1. play games (raw engine vs SF), recording every position AND move played
2. Stockfish grades every position; flag our blunders (>= swing, before-ceiling)
3. for each blunder, ask Stockfish for the best move, then compare the raw
   eval BREAKDOWN of [position after our move] vs [position after SF's best]
   -> the per-term deltas show exactly which eval term mis-ranked the two
   futures (or which permanently-zero term is missing).
Output: a human-readable report, worst blunders first.
"""
import argparse
import json
import os
import subprocess
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_BIN = os.path.join(REPO, "tools", "uci_engine_raw.exe")
EVAL_CLI = os.path.join(REPO, "tools", "raw_eval_cli.exe")


def breakdown(fen):
    out = subprocess.run([EVAL_CLI, fen], capture_output=True, text=True)
    d = {}
    for line in out.stdout.strip().splitlines():
        k, v = line.rsplit(" ", 1)
        d[k] = int(v)
    return d


def rollout(fen, engine, plies, ms):
    """let the raw engine play both sides for a few plies -> its expected future"""
    board = chess.Board(fen)
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        mv = engine.play(board, chess.engine.Limit(time=ms / 1000), game=object()).move
        if mv is None:
            break
        board.push(mv)
    return board.fen()


def play_games(n_games, our_ms, sf_ms, sf_elo, max_plies):
    env = dict(os.environ)
    env["RAW_WEIGHT"] = "1"
    our = chess.engine.SimpleEngine.popen_uci(RAW_BIN, env=env)
    sf = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
    games = []
    for g in range(n_games):
        we_white = (g % 2 == 0)
        board = chess.Board()
        plies = []  # (fen_before, we_move, uci_move)
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            we_move = (board.turn == chess.WHITE) == we_white
            fen = board.fen()
            eng, lim = (our, chess.engine.Limit(time=our_ms / 1000)) if we_move \
                else (sf, chess.engine.Limit(time=sf_ms / 1000))
            mv = eng.play(board, lim, game=object()).move
            if mv is None:
                break
            plies.append((fen, we_move, mv.uci()))
            board.push(mv)
        res = board.result(claim_draw=True)
        our_pov = 1.0 if (res == "1-0") == we_white and res != "1/2-1/2" else (0.5 if res == "1/2-1/2" else 0.0)
        games.append({"plies": plies, "we_white": we_white, "result": our_pov})
        print(f"  game {g+1}/{n_games}: our_result={our_pov}", flush=True)
    our.quit()
    sf.quit()
    return games


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--sf-elo", type=int, default=2200)
    p.add_argument("--our-ms", type=int, default=500)
    p.add_argument("--sf-ms", type=int, default=500)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--grade-depth", type=int, default=12)
    p.add_argument("--analyse-depth", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--swing-thresh", type=int, default=120)
    p.add_argument("--before-ceiling", type=int, default=600)
    p.add_argument("--top", type=int, default=15, help="report the N worst blunders")
    p.add_argument("--out", default="tools/elo_logs/raw_diagnosis.txt")
    args = p.parse_args()

    print(f"playing {args.games} games vs SF@{args.sf_elo}...", flush=True)
    games = play_games(args.games, args.our_ms, args.sf_ms, args.sf_elo, args.max_plies)
    w = sum(1 for g in games if g["result"] == 1.0)
    d = sum(1 for g in games if g["result"] == 0.5)
    l = sum(1 for g in games if g["result"] == 0.0)
    print(f"record: {w}W/{d}D/{l}L", flush=True)

    fens = [fen for g in games for fen, _we, _mv in g["plies"]]
    fens = list(dict.fromkeys(fens))
    print(f"grading {len(fens)} positions at depth {args.grade_depth}...", flush=True)
    graded = label_positions_parallel(fens, DEFAULT_STOCKFISH_PATH, args.grade_depth, args.workers)
    by_fen = {r["fen"]: (r["cp"] if r["cp"] is not None
                          else (2000 if r["mate"] and r["mate"] > 0 else -2000))
              for r in graded}

    blunders = []  # (swing, fen_before, our_move, we_white)
    for g in games:
        plies = g["plies"]
        for i in range(len(plies) - 1):
            fen, we_move, mv = plies[i]
            if not we_move:
                continue
            nfen = plies[i + 1][0]
            if fen not in by_fen or nfen not in by_fen:
                continue
            sgn = 1 if g["we_white"] else -1
            before = sgn * max(-2000, min(2000, by_fen[fen]))
            after = sgn * max(-2000, min(2000, by_fen[nfen]))
            if abs(before) > args.before_ceiling:
                continue
            swing = before - after
            if swing >= args.swing_thresh:
                blunders.append((swing, fen, mv, g["we_white"]))

    blunders.sort(reverse=True)
    blunders = blunders[:args.top]
    print(f"analysing {len(blunders)} worst blunders at depth {args.analyse_depth}...", flush=True)

    sf = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH_PATH)
    sf.configure({"Threads": args.workers, "Hash": 256})
    renv = dict(os.environ); renv["RAW_WEIGHT"] = "1"
    roll_eng = chess.engine.SimpleEngine.popen_uci(RAW_BIN, env=renv)

    lines = []
    for swing, fen, our_mv, we_white in blunders:
        board = chess.Board(fen)
        info = sf.analyse(board, chess.engine.Limit(depth=args.analyse_depth))
        best_mv = info["pv"][0].uci() if info.get("pv") else "?"
        sf_best_score = info["score"].white().score(mate_score=2000)
        if best_mv == our_mv:
            continue  # deeper analysis says our move was fine after all

        b_ours = board.copy(); b_ours.push_uci(our_mv)
        b_best = board.copy(); b_best.push_uci(best_mv)
        info_ours = sf.analyse(b_ours, chess.engine.Limit(depth=args.analyse_depth - 2))
        sf_after_ours = info_ours["score"].white().score(mate_score=2000)

        bd_ours = breakdown(b_ours.fen())
        bd_best = breakdown(b_best.fen())
        leaf_ours_fen = rollout(b_ours.fen(), roll_eng, 4, 150)
        leaf_best_fen = rollout(b_best.fen(), roll_eng, 4, 150)
        ld_ours = breakdown(leaf_ours_fen)
        ld_best = breakdown(leaf_best_fen)
        pov = 1 if we_white else -1
        deltas = {k: pov * (bd_ours.get(k, 0) - bd_best.get(k, 0))
                  for k in bd_ours if k not in ("phase",)}
        worst_terms = sorted(deltas.items(), key=lambda kv: kv[1])[:5]

        lines.append(f"\n=== swing {swing}cp (we are {'White' if we_white else 'Black'}) ===")
        lines.append(f"pos: {fen}")
        lines.append(f"played {our_mv} | SF best {best_mv} (SF says {sf_best_score}cp before, "
                     f"{sf_after_ours}cp after our move, white-POV)")
        lines.append(f"raw total after ours {bd_ours['total']} vs after best {bd_best['total']} "
                     f"(our-POV pref for our move: {pov * (bd_ours['total'] - bd_best['total'])}cp)")
        lines.append("terms favoring our (bad) move vs best (our-POV, most-negative = "
                     "term that failed to punish our choice... positive = term that "
                     "wrongly PREFERRED our move):")
        top_pos = sorted(deltas.items(), key=lambda kv: -kv[1])[:5]
        lines.append("  overrated-our-move: " + ", ".join(f"{k} {v:+d}" for k, v in top_pos if v > 0))
        lines.append("  underrated-best:    " + ", ".join(f"{k} {v:+d}" for k, v in worst_terms if v < 0))
        ldeltas = {k: pov * (ld_ours.get(k, 0) - ld_best.get(k, 0))
                   for k in ld_ours if k not in ("phase",)}
        lines.append(f"LEAF (4-ply raw rollout): our-line total {pov*ld_ours['total']} vs "
                     f"best-line {pov*ld_best['total']} (our-POV)")
        lp = sorted(ldeltas.items(), key=lambda kv: -kv[1])[:4]
        ln = sorted(ldeltas.items(), key=lambda kv: kv[1])[:4]
        lines.append("  leaf overrated-our-line: " + ", ".join(f"{k} {v:+d}" for k, v in lp if v > 0))
        lines.append("  leaf underrated-best:    " + ", ".join(f"{k} {v:+d}" for k, v in ln if v < 0))
        lines.append(f"  leaf fens: ours={leaf_ours_fen} | best={leaf_best_fen}")

    roll_eng.quit()
    sf.quit()
    report = "\n".join(lines) if lines else "no confirmed blunders at analyse depth"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report + "\n")
    print(report, flush=True)
    print(f"\nreport written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
