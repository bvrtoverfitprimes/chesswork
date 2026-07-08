import argparse
import json
import os
import random
import subprocess
import sys
import time

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLI = os.environ.get("HL_CLI_OVERRIDE", os.path.join(REPO, "tools", "bestmove_cli.exe"))
SF_PATH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                        "stockfish-windows-x86-64-avx2.exe")

MATE_CP = 100000


def sample_fens(path, n, seed, scan_limit=200_000):
    rng = random.Random(seed)
    reservoir = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= scan_limit:
                break
            row = json.loads(line)
            fen = row["fen"]
            board = chess.Board(fen)
            if board.is_game_over() or len(list(board.legal_moves)) < 2:
                continue
            if len(reservoir) < n:
                reservoir.append(fen)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = fen
    rng.shuffle(reservoir)
    return reservoir


def our_move(engine_label, fen, time_ms):
    out = subprocess.run([CLI, engine_label, fen, str(time_ms)], capture_output=True, text=True, timeout=60)
    return out.stdout.strip()


def stm_score(info, stm_color):
    return info["score"].pov(stm_color).score(mate_score=MATE_CP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="human")
    parser.add_argument("--our-time-ms", type=int, default=500)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ground-truth-depth", type=int, default=16)
    parser.add_argument("--positions", default=os.path.join(REPO, "training", "data", "raw", "lichess_6m.jsonl"))
    parser.add_argument("--fixed-fens", default=os.path.join(REPO, "training", "eval", "move_quality_fens.json"))
    parser.add_argument("--regenerate", action="store_true", help="resample fresh positions instead of using the fixed set")
    args = parser.parse_args()

    if args.regenerate or not os.path.exists(args.fixed_fens):
        print(f"sampling {args.n} positions (seed={args.seed})...", flush=True)
        fens = sample_fens(args.positions, args.n, args.seed)
        with open(args.fixed_fens, "w") as f:
            json.dump(fens, f, indent=2)
    else:
        with open(args.fixed_fens) as f:
            fens = json.load(f)[:args.n]
        print(f"using {len(fens)} fixed positions from {args.fixed_fens}", flush=True)

    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"Threads": 1})

    losses = []
    matches = 0
    t0 = time.time()
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        stm = board.turn

        info_best = sf.analyse(board, chess.engine.Limit(depth=args.ground_truth_depth))
        best_move = info_best["pv"][0]
        best_score = stm_score(info_best, stm)

        move_uci = our_move(args.engine, fen, args.our_time_ms)
        try:
            move = chess.Move.from_uci(move_uci)
            board.push(move)
        except Exception:
            print(f"pos {i+1}: ENGINE ERROR (got {move_uci!r})", flush=True)
            losses.append(MATE_CP)
            continue

        info_ours = sf.analyse(board, chess.engine.Limit(depth=args.ground_truth_depth))
        our_score = stm_score(info_ours, stm)
        loss = max(0, best_score - our_score)
        losses.append(loss)
        if move == best_move:
            matches += 1

        print(f"pos {i+1}/{len(fens)}: best={best_move.uci()}(cp={best_score}) "
              f"ours={move_uci}(cp={our_score}) loss={loss}", flush=True)

    sf.quit()
    elapsed = time.time() - t0
    avg_loss = sum(losses) / len(losses) if losses else 0
    print(f"\n{args.engine} (time={args.our_time_ms}ms): "
          f"avg_centipawn_loss={avg_loss:.1f} exact_match_rate={matches}/{len(fens)} "
          f"({100.0*matches/len(fens):.0f}%) over {len(fens)} positions in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
