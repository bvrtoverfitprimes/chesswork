"""Fast feature-ablation by DECISION IMPACT (seconds, no games). For a sample of
positions, generate every legal move, eval all children with each eval feature
ON vs OFF, and count how often turning a feature off changes which move has the
best (depth-1) eval. A feature that rarely/never changes the chosen move is
fluff (simplify); one that changes many decisions matters. Also reports each
feature's mean |contribution| in cp. Uses raw_eval_tune 'batch' mode."""
import os
import subprocess
import sys

import chess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "tools", "raw_eval_tune.exe")
POS = os.path.join(REPO, "training", "tune", "tuning_positions.txt")
NAMES = ["mobility(N/R/Q)", "bishop_mobility", "king_safety", "threats",
         "passed_pawns", "pawn_structure", "piece_quality", "rook_files",
         "bishop_pair", "endgame_king", "center_control", "outposts"]
NSAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def batch_eval(fens, scales):
    env = dict(os.environ); env["RAW_WEIGHT"] = "1"
    env["RAW_TUNE"] = " ".join(str(x) for x in scales)
    p = subprocess.run([CLI, "batch"], input="\n".join(fens) + "\n",
                       capture_output=True, text=True, env=env)
    return [int(x) for x in p.stdout.split()]


def main():
    positions = []
    for line in open(POS):
        line = line.rstrip("\n")
        if line:
            positions.append(line.split("\t")[0])
    positions = positions[:NSAMPLE]

    # build child list: for each position, all legal-move children (+ whose turn)
    child_fens = []
    groups = []  # (start, end, white_to_move)
    for fen in positions:
        b = chess.Board(fen)
        wtm = b.turn == chess.WHITE
        start = len(child_fens)
        for mv in b.legal_moves:
            b.push(mv)
            child_fens.append(b.fen())
            b.pop()
        if len(child_fens) > start:
            groups.append((start, len(child_fens), wtm))
    print(f"{len(groups)} positions, {len(child_fens)} child evals x 13 configs", flush=True)

    ident = [128] * 12
    full = batch_eval(child_fens, ident)

    def best_moves(evals):
        out = []
        for start, end, wtm in groups:
            seg = evals[start:end]
            # our best move maximizes our-POV eval (white-rel if white, else negate)
            best_i = max(range(len(seg)), key=lambda k: seg[k] if wtm else -seg[k])
            out.append(best_i)
        return out

    full_best = best_moves(full)

    results = []
    for i in range(12):
        sc = ident[:]; sc[i] = 0
        ev = batch_eval(child_fens, sc)
        off_best = best_moves(ev)
        changed = sum(1 for a, b in zip(full_best, off_best) if a != b)
        # mean |contribution| across all children in cp
        contrib = sum(abs(f - e) for f, e in zip(full, ev)) / len(full)
        pct = 100.0 * changed / len(groups)
        results.append((NAMES[i], pct, contrib, changed))
        print(f"[{i:2d}] {NAMES[i]:18s}  changes {pct:5.1f}% of moves  "
              f"mean|contrib| {contrib:5.1f}cp", flush=True)

    print("\n=== RANKED by decision impact (most moves changed first) ===", flush=True)
    for name, pct, contrib, changed in sorted(results, key=lambda x: -x[1]):
        tag = "FLUFF (simplify)" if pct < 1.0 else ("minor" if pct < 5 else "IMPORTANT")
        print(f"  {name:18s}  {pct:5.1f}% moves  {contrib:5.1f}cp  [{tag}]", flush=True)


if __name__ == "__main__":
    main()
