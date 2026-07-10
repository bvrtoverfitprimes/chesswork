"""Manual fix loop per user's method:
  play 2 recorded games vs calibrated SF -> strong SF grades every position ->
  report exact worst-blunder positions -> (human/operator fixes eval) ->
  verify the fix on those exact positions AND on every previously-fixed
  position (cumulative regression suite) -> repeat.

Modes:
  play    : play 2 games (one White, one Black) vs --sf-elo, grade, report top
            blunders, append them to the regression file
  verify  : for every regression entry, ask the current engine for its move at
            --ms; report FIXED (plays SF-best or eval-swing gone) vs STILL-BAD
"""
import argparse
import json
import os
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_BIN = os.path.join(REPO, "tools", "uci_engine_raw.exe")
REG_FILE = os.path.join(REPO, "training", "data", "raw", "fix_regression.jsonl")


def open_raw(ms_unused=None):
    env = dict(os.environ)
    env["RAW_WEIGHT"] = "1"
    return chess.engine.SimpleEngine.popen_uci(RAW_BIN, env=env)


def play_two(sf_elo, our_ms, sf_ms, max_plies, game_offset=0):
    our = open_raw()
    sf = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})
    games = []
    for g in range(2):
        we_white = ((g + game_offset) % 2 == 0)
        board = chess.Board()
        plies = []
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
        our_pov = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == we_white else 0.0)
        games.append({"plies": plies, "we_white": we_white, "result": our_pov})
        print(f"game {g+1}/2 ({'White' if we_white else 'Black'}): our_result={our_pov}", flush=True)
    our.quit()
    sf.quit()
    return games


def find_blunders(games, grade_depth, analyse_depth, workers, swing_thresh, before_ceiling, top):
    fens = list(dict.fromkeys(f for g in games for f, _w, _m in g["plies"]))
    print(f"grading {len(fens)} positions at depth {grade_depth}...", flush=True)
    graded = label_positions_parallel(fens, DEFAULT_STOCKFISH_PATH, grade_depth, workers)
    by_fen = {r["fen"]: (r["cp"] if r["cp"] is not None
                          else (2000 if r["mate"] and r["mate"] > 0 else -2000)) for r in graded}
    cands = []
    for g in games:
        plies = g["plies"]
        sgn = 1 if g["we_white"] else -1
        for i in range(len(plies) - 1):
            fen, we, mv = plies[i]
            if not we or fen not in by_fen or plies[i+1][0] not in by_fen:
                continue
            before = sgn * max(-2000, min(2000, by_fen[fen]))
            after = sgn * max(-2000, min(2000, by_fen[plies[i+1][0]]))
            if abs(before) <= before_ceiling and before - after >= swing_thresh:
                cands.append((before - after, fen, mv, g["we_white"]))
    cands.sort(reverse=True)

    sf = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH_PATH)
    sf.configure({"Threads": workers, "Hash": 256})
    out = []
    for swing, fen, our_mv, we_white in cands[:top]:
        board = chess.Board(fen)
        info = sf.analyse(board, chess.engine.Limit(depth=analyse_depth))
        best = info["pv"][0].uci() if info.get("pv") else "?"
        if best == our_mv:
            continue
        out.append({"fen": fen, "our_mv": our_mv, "best_mv": best, "swing": swing,
                    "we_white": we_white,
                    "sf_eval_w": info["score"].white().score(mate_score=2000)})
    sf.quit()
    return out


def verify(entries, ms):
    our = open_raw()
    fixed = still = 0
    for e in entries:
        board = chess.Board(e["fen"])
        mv = our.play(board, chess.engine.Limit(time=ms / 1000), game=object()).move.uci()
        ok = (mv == e["best_mv"]) or (mv != e["our_mv"])
        status = "PLAYS-BEST" if mv == e["best_mv"] else ("CHANGED" if mv != e["our_mv"] else "STILL-BAD")
        if mv != e["our_mv"]:
            fixed += 1
        else:
            still += 1
        print(f"[{status}] swing{e['swing']} {e['fen']}  played {mv} (was {e['our_mv']}, best {e['best_mv']})",
              flush=True)
    our.quit()
    print(f"\n{fixed} changed-or-best / {still} still playing the blunder of {fixed+still}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["play", "verify"])
    p.add_argument("--sf-elo", type=int, default=2300)
    p.add_argument("--our-ms", type=int, default=500)
    p.add_argument("--sf-ms", type=int, default=500)
    p.add_argument("--ms", type=int, default=500, help="verify movetime")
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--grade-depth", type=int, default=12)
    p.add_argument("--analyse-depth", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--swing-thresh", type=int, default=120)
    p.add_argument("--before-ceiling", type=int, default=600)
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--game-offset", type=int, default=0)
    args = p.parse_args()

    if args.mode == "play":
        games = play_two(args.sf_elo, args.our_ms, args.sf_ms, args.max_plies, args.game_offset)
        blunders = find_blunders(games, args.grade_depth, args.analyse_depth, args.workers,
                                  args.swing_thresh, args.before_ceiling, args.top)
        if not blunders:
            print("no confirmed blunders this pair", flush=True)
            return
        with open(REG_FILE, "a") as f:
            for b in blunders:
                f.write(json.dumps(b) + "\n")
        for b in blunders:
            print(json.dumps(b), flush=True)
        print(f"{len(blunders)} blunders appended to regression file "
              f"({sum(1 for _ in open(REG_FILE))} total)", flush=True)
    else:
        entries = [json.loads(l) for l in open(REG_FILE) if l.strip()]
        verify(entries, args.ms)


if __name__ == "__main__":
    main()
