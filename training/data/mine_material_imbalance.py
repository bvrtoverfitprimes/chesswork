"""Targeted weak-point mining for material-imbalance magnitude calibration.

Motivation (see WORKLOG §24/§25 and tactical_sanity.py): the eval has historically
under-scored decisive material advantages, especially for Black. As of this
session, tactical_sanity shows a real, large color asymmetry on the simplest
possible case -- bare kings, one side up a queen:
  White up a queen: +575cp (passes, needs > 500)
  Black up a queen:  -453cp (FAILS, needs < -500)
Both sides have exactly the mirrored material, so a ~120cp gap is a genuine
calibration defect, not just "hard position needs more data of this shape."

This generates a batch of clearly-decisive, color-balanced material-imbalance
positions (bare-king endgames + full-army-minus-a-piece middlegames, one side
up a queen/rook/minor, in equal number for White-up and Black-up), deep-grades
them with Stockfish, and emits a jsonl in the same {fen, cp, mate, depth}
format mine_blunders.py produces -- ready to be `cat`-combined with the mined
blunder sets and used as an --aux-dataset for a pure-WDL fine-tune (same
discipline as the blunder-mining loop: no loss-function tricks, just better/
more-relevant labels, judged by A/B games before adoption).
"""
import argparse
import json
import os
import random
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.stockfish_teacher import label_positions_parallel, DEFAULT_STOCKFISH_PATH


def _random_square(rng, exclude):
    while True:
        sq = rng.randrange(64)
        if sq not in exclude:
            return sq


def _kings_ok(k1, k2):
    return chess.square_distance(k1, k2) > 1


def bare_king_imbalance_fen(rng, extra_piece_type, strong_side):
    board = chess.Board.empty()
    used = set()
    wk = _random_square(rng, used)
    used.add(wk)
    bk = _random_square(rng, used)
    while not _kings_ok(wk, bk):
        bk = _random_square(rng, used)
    used.add(bk)
    board.set_piece_at(wk, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(bk, chess.Piece(chess.KING, chess.BLACK))

    sq = _random_square(rng, used)
    tries = 0
    while extra_piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7) and tries < 20:
        sq = _random_square(rng, used)
        tries += 1
    used.add(sq)
    board.set_piece_at(sq, chess.Piece(extra_piece_type, strong_side))
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid() or board.is_checkmate() or board.is_stalemate():
        return None
    return board.fen()


_MIDGAME_BASES = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPPQPPP/R1B1KB1R w KQkq - 4 5",
    "r2qkbnr/ppp1pppp/2np4/8/3PP3/5N2/PPP2PPP/RNBQKB1R w KQkq - 2 4",
    "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 2 6",
]


def full_army_minus_piece_fen(rng, piece_type, removed_side):
    base = rng.choice(_MIDGAME_BASES)
    board = chess.Board(base)
    candidates = [
        sq for sq in chess.SQUARES
        if board.piece_at(sq) is not None
        and board.piece_at(sq).piece_type == piece_type
        and board.piece_at(sq).color == removed_side
    ]
    if not candidates:
        return None
    board.remove_piece_at(rng.choice(candidates))
    board.castling_rights = chess.BB_EMPTY
    board.turn = rng.choice([chess.WHITE, chess.BLACK])
    if not board.is_valid():
        return None
    return board.fen()


def generate(rng, n_per_bucket):
    fens = []
    for piece_type in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
        for strong_side in (chess.WHITE, chess.BLACK):
            made = 0
            attempts = 0
            while made < n_per_bucket and attempts < n_per_bucket * 8:
                attempts += 1
                f = bare_king_imbalance_fen(rng, piece_type, strong_side)
                if f:
                    fens.append(f)
                    made += 1
    for piece_type in (chess.QUEEN, chess.ROOK):
        for removed_side in (chess.WHITE, chess.BLACK):
            made = 0
            attempts = 0
            while made < n_per_bucket and attempts < n_per_bucket * 8:
                attempts += 1
                f = full_army_minus_piece_fen(rng, piece_type, removed_side)
                if f:
                    fens.append(f)
                    made += 1
    return list(dict.fromkeys(fens))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-bucket", type=int, default=40)
    p.add_argument("--deep-depth", type=int, default=16)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="training/data/raw/material_imbalance_labeled.jsonl")
    args = p.parse_args()
    rng = random.Random(args.seed)

    fens = generate(rng, args.n_per_bucket)
    print(f"generated {len(fens)} color-balanced material-imbalance positions; "
          f"grading at depth {args.deep_depth}...", flush=True)
    labeled = label_positions_parallel(fens, DEFAULT_STOCKFISH_PATH, args.deep_depth, args.workers)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for r in labeled:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(labeled)} labeled positions to {args.out}", flush=True)


if __name__ == "__main__":
    main()
