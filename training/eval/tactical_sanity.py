import argparse
import os
import subprocess

import chess

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _remove_piece_fen(base_fen, square_name, expected_piece_type):
    board = chess.Board(base_fen)
    sq = chess.parse_square(square_name)
    piece = board.piece_at(sq)
    assert piece is not None and piece.piece_type == expected_piece_type, \
        f"expected {expected_piece_type} at {square_name} in {base_fen}, found {piece}"
    board.remove_piece_at(sq)
    board.castling_rights = chess.BB_EMPTY
    assert board.is_valid()
    return board.fen()


_MIDGAME_BASE = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPPQPPP/R1B1KB1R w KQkq - 4 5"


def eval_cp(eval_cli, weights, fen):
    out = subprocess.run([eval_cli, fen, weights], capture_output=True, text=True)
    return float(out.stdout.strip())


def best_move(bestmove_cli, fen, time_ms=500):
    out = subprocess.run([bestmove_cli, "human", fen, str(time_ms)], capture_output=True, text=True)
    return out.stdout.strip()


MATERIAL_CASES = [
    ("white up a queen, bare kings", "4k3/8/8/8/8/8/8/3QK3 w - - 0 1", lambda cp: cp > 500),
    ("black up a queen, bare kings", "3qk3/8/8/8/8/8/8/4K3 w - - 0 1", lambda cp: cp < -500),
    ("white up a queen, full middlegame army",
     _remove_piece_fen(_MIDGAME_BASE, "d8", chess.QUEEN), lambda cp: cp > 400),
    ("black up a rook, full middlegame army",
     _remove_piece_fen(_MIDGAME_BASE, "h1", chess.ROOK), lambda cp: cp < -300),
    ("start position roughly balanced", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     lambda cp: abs(cp) < 150),
]

TACTICAL_CASES = [
    ("back-rank mate in 1", "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
    ("hanging queen capture available", "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1", "d1d5"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-cli", default=os.path.join(REPO_ROOT, "tools", "eval_cli.exe"))
    parser.add_argument("--bestmove-cli", default=os.path.join(REPO_ROOT, "tools", "bestmove_cli.exe"))
    parser.add_argument("--weights", default=os.path.join(REPO_ROOT, "engine", "limit", "nnue_weights.bin"))
    parser.add_argument("--time-ms", type=int, default=500)
    args = parser.parse_args()

    failures = 0

    for label, fen, predicate in MATERIAL_CASES:
        cp = eval_cp(args.eval_cli, args.weights, fen)
        ok = predicate(cp)
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {label} (cp={cp:.1f})")
        if not ok:
            failures += 1

    for label, fen, expected_move in TACTICAL_CASES:
        move = best_move(args.bestmove_cli, fen, args.time_ms)
        ok = move == expected_move
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {label} (got {move}, expected {expected_move})")
        if not ok:
            failures += 1

    print(f"\n{len(MATERIAL_CASES) + len(TACTICAL_CASES) - failures}/{len(MATERIAL_CASES) + len(TACTICAL_CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
