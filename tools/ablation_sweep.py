"""Feature-ablation / simplification study. For each eval category, play a fixed
match of champion-with-that-category-OFF (scale 0) vs the FULL champion (identity),
same tunebase binary. A score well below 50% => the feature MATTERS (keep). A
score near/above 50% => the feature is fluff (candidate to remove/simplify).
Ranks categories by importance. All games start from the true start position.
"""
import argparse
import math
import os

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(REPO, "tools", "uci_engine_tunebase.exe")
NAMES = ["mobility_minor_major", "bishop_mobility", "king_safety", "threats",
         "passed_pawns", "pawn_structure", "piece_quality", "rook_files",
         "bishop_pair", "endgame_king", "center_control", "outposts"]


def open_eng(off_index):
    env = dict(os.environ)
    env["RAW_WEIGHT"] = "1"
    if off_index is not None:
        scales = [128] * 12
        scales[off_index] = 0
        env["RAW_TUNE"] = " ".join(str(x) for x in scales)
    elif "RAW_TUNE" in env:
        del env["RAW_TUNE"]
    return chess.engine.SimpleEngine.popen_uci(BIN, env=env)


def play(a, b, ms, a_white, max_plies=240):
    board = chess.Board()
    lim = chess.engine.Limit(time=ms / 1000)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = a if (board.turn == chess.WHITE) == a_white else b
        mv = eng.play(board, lim, game=object()).move
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
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--ms", type=int, default=400)
    args = ap.parse_args()

    results = []
    full = open_eng(None)  # baseline B = full champion
    for i in range(12):
        off = open_eng(i)
        score = 0.0
        w = d = l = 0
        for g in range(args.games):
            r = play(off, full, args.ms, g % 2 == 0)
            score += r
            if r == 1.0:
                w += 1
            elif r == 0.5:
                d += 1
            else:
                l += 1
        off.quit()
        pct = 100.0 * score / args.games
        e = elo(score / args.games)
        results.append((NAMES[i], pct, e, w, d, l))
        print(f"[{i:2d}] {NAMES[i]:22s} OFF: {pct:5.1f}% ({w}W/{d}D/{l}L)  "
              f"=> feature worth ~{-e:+.0f} Elo", flush=True)
    full.quit()

    print("\n=== RANKED by importance (most Elo lost when removed first) ===", flush=True)
    for name, pct, e, w, d, l in sorted(results, key=lambda x: x[2]):
        tag = "IMPORTANT" if pct < 43 else ("fluff?" if pct >= 48 else "minor")
        print(f"  {name:22s} off={pct:5.1f}%  worth ~{-e:+.0f} Elo  [{tag}]", flush=True)


if __name__ == "__main__":
    main()
