"""Hard-example mining: play our engine vs Stockfish, find positions where our
engine BLUNDERED (Stockfish eval swings sharply against us on our own move),
perturb each blunder position into ~N near-neighbours (one random piece changed
on either side), grade everything with a deep Stockfish, and emit a training
jsonl focused on the model's actual weak points.

Stages (each can be skipped if its output already exists via --skip-play):
  1. play games (our uci_engine vs Stockfish), record every ply
  2. grade every position of games we did NOT win, detect our blunders
  3. perturb each blunder position; grade blunders + perturbations at deep depth
  4. write {fen, cp, mate, depth} jsonl ready for build_dataset.py
"""
import argparse
import json
import os
import random
import sys

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUR = os.path.join(REPO, "tools", "uci_engine.exe")


def our_advantage_cp(white_rel_cp, we_are_white):
    """Convert a White-relative cp score to our-side-relative (clamped for mates)."""
    v = max(-2000, min(2000, white_rel_cp))
    return v if we_are_white else -v


def play_games(n_games, our_ms, sf_ms, sf_elo, threads, max_plies):
    """Return list of games: each is dict(plies=[fen_before_move...], we_white, result_our_pov)."""
    os.environ.setdefault("HL_WEIGHTS", "engine/human_limit/nnue_weights.bin")
    our = chess.engine.SimpleEngine.popen_uci(OUR)
    if threads > 1:
        try:
            our.configure({"Threads": threads})
        except Exception:
            pass
    sf = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH_PATH)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo, "Threads": 1})

    games = []
    for g in range(n_games):
        we_white = (g % 2 == 0)
        board = chess.Board()
        plies = []  # (fen_before, we_moved)
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            we_move = (board.turn == chess.WHITE) == we_white
            plies.append((board.fen(), we_move))
            eng, lim = (our, chess.engine.Limit(time=our_ms / 1000)) if we_move \
                else (sf, chess.engine.Limit(time=sf_ms / 1000))
            mv = eng.play(board, lim, game=object()).move
            if mv is None:
                break
            board.push(mv)
        res = board.result(claim_draw=True)
        if res == "1-0":
            our_pov = 1.0 if we_white else 0.0
        elif res == "0-1":
            our_pov = 0.0 if we_white else 1.0
        else:
            our_pov = 0.5
        games.append({"plies": plies, "we_white": we_white, "result": our_pov})
        print(f"  played game {g+1}/{n_games}: our_result={our_pov}", flush=True)
    our.quit()
    sf.quit()
    return games


def detect_blunders(games, grade_depth, workers, swing_thresh, skip_won):
    """Grade every position of non-won games; return FENs of positions where OUR
    move dropped our eval by >= swing_thresh (i.e. our blunder input positions)."""
    # gather all FENs to grade, remembering their (game, ply) location
    index = []  # (gi, pi)
    fens = []
    for gi, game in enumerate(games):
        if skip_won and game["result"] == 1.0:
            continue
        for pi, (fen, _we) in enumerate(game["plies"]):
            index.append((gi, pi))
            fens.append(fen)
    if not fens:
        return []
    print(f"grading {len(fens)} game positions at depth {grade_depth}...", flush=True)
    graded = label_positions_parallel(fens, DEFAULT_STOCKFISH_PATH, grade_depth, workers)
    # label_positions_parallel does not preserve order across workers -> map by fen
    by_fen = {}
    for r in graded:
        cp = r["cp"] if r["cp"] is not None else (2000 if r["mate"] and r["mate"] > 0 else -2000)
        by_fen[r["fen"]] = cp

    blunders = []
    for gi, game in enumerate(games):
        if skip_won and game["result"] == 1.0:
            continue
        plies = game["plies"]
        for pi in range(len(plies) - 1):
            fen_before, we_move = plies[pi]
            if not we_move:
                continue
            fen_after = plies[pi + 1][0]
            if fen_before not in by_fen or fen_after not in by_fen:
                continue
            adv_before = our_advantage_cp(by_fen[fen_before], game["we_white"])
            adv_after = our_advantage_cp(by_fen[fen_after], game["we_white"])
            if adv_before - adv_after >= swing_thresh:
                blunders.append(fen_before)
    print(f"detected {len(blunders)} blunder positions "
          f"(eval dropped >= {swing_thresh}cp on our move)", flush=True)
    return blunders


def perturb(fen, rng, n_variants, max_tries=60):
    """Generate up to n_variants legal near-neighbours: remove / add / relocate one
    non-king piece. Returns list of FENs (side-to-move preserved)."""
    base = chess.Board(fen)
    out = []
    seen = set()
    tries = 0
    while len(out) < n_variants and tries < max_tries:
        tries += 1
        b = base.copy()
        op = rng.choice(["remove", "add", "move"])
        piece_squares = [sq for sq in chess.SQUARES
                         if b.piece_at(sq) and b.piece_at(sq).piece_type != chess.KING]
        empties = [sq for sq in chess.SQUARES if b.piece_at(sq) is None]
        try:
            if op == "remove" and piece_squares:
                b.remove_piece_at(rng.choice(piece_squares))
            elif op == "add" and empties:
                sq = rng.choice(empties)
                pt = rng.choice([chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN])
                col = rng.choice([chess.WHITE, chess.BLACK])
                if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                    continue
                b.set_piece_at(sq, chess.Piece(pt, col))
            elif op == "move" and piece_squares and empties:
                src = rng.choice(piece_squares)
                dst = rng.choice(empties)
                pc = b.piece_at(src)
                if pc.piece_type == chess.PAWN and chess.square_rank(dst) in (0, 7):
                    continue
                b.remove_piece_at(src)
                b.set_piece_at(dst, pc)
            else:
                continue
        except Exception:
            continue
        b.clear_stack()
        if not b.is_valid():
            continue
        f = b.fen()
        if f in seen or f == fen:
            continue
        seen.add(f)
        out.append(f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--sf-elo", type=int, default=2200)
    p.add_argument("--our-ms", type=int, default=500)
    p.add_argument("--sf-ms", type=int, default=500)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--grade-depth", type=int, default=12, help="depth for blunder detection")
    p.add_argument("--deep-depth", type=int, default=16, help="depth for final training labels")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--swing-thresh", type=int, default=150)
    p.add_argument("--variants", type=int, default=10)
    # Mine our blunders from ALL games by default: a move that dropped our eval is
    # a weak point even in games we won/drew (the opponent just didn't punish it).
    p.add_argument("--skip-won", dest="skip_won", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--games-cache", default="training/data/raw/blunder_games.json")
    p.add_argument("--out", default="training/data/raw/blunder_labeled.jsonl")
    args = p.parse_args()
    rng = random.Random(args.seed)

    if os.path.exists(args.games_cache):
        print(f"loading cached games from {args.games_cache}", flush=True)
        with open(args.games_cache) as f:
            games = json.load(f)
    else:
        print(f"playing {args.games} games vs Stockfish@{args.sf_elo}...", flush=True)
        games = play_games(args.games, args.our_ms, args.sf_ms, args.sf_elo, args.threads, args.max_plies)
        os.makedirs(os.path.dirname(args.games_cache), exist_ok=True)
        with open(args.games_cache, "w") as f:
            json.dump(games, f)

    won = sum(1 for g in games if g["result"] == 1.0)
    drew = sum(1 for g in games if g["result"] == 0.5)
    lost = sum(1 for g in games if g["result"] == 0.0)
    print(f"games: {won}W / {drew}D / {lost}L", flush=True)

    blunders = detect_blunders(games, args.grade_depth, args.workers, args.swing_thresh, args.skip_won)
    if not blunders:
        print("no blunders found; nothing to train on.", flush=True)
        return

    # dedupe blunder positions, then perturb each
    blunders = list(dict.fromkeys(blunders))
    all_fens = list(blunders)
    for fen in blunders:
        all_fens.extend(perturb(fen, rng, args.variants))
    all_fens = list(dict.fromkeys(all_fens))
    print(f"mined {len(blunders)} blunders -> {len(all_fens)} positions "
          f"(with {args.variants} perturbations each); grading at depth {args.deep_depth}...", flush=True)

    labeled = label_positions_parallel(all_fens, DEFAULT_STOCKFISH_PATH, args.deep_depth, args.workers)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in labeled:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(labeled)} deeply-graded weak-point positions to {args.out}", flush=True)


if __name__ == "__main__":
    main()
