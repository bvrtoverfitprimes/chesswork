"""Play a few quick games vs Stockfish with the CURRENT production weights, find
where our own move caused the biggest Stockfish-eval swings (the same detection
method as training/data/mine_blunders.py), and for the top examples print:
  - Stockfish's deep evaluation of the position before our move
  - Stockfish's suggested best move there
  - the move WE actually played
  - Stockfish's evaluation after our move (showing the damage)
  - OUR ENGINE's own raw static evaluation of the position (via eval_cli) --
    this is the number that shows the eval discrepancy behind the blunder.
"""
import os
import subprocess
import sys

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "training"))
from data.mine_blunders import play_games, our_advantage_cp
from data.stockfish_teacher import DEFAULT_STOCKFISH_PATH

EVAL_CLI = os.path.join(REPO, "tools", "eval_cli.exe")


def our_eval(fen):
    out = subprocess.run([EVAL_CLI, fen], capture_output=True, text=True, timeout=30)
    return float(out.stdout.strip())


def main():
    n_games = 6
    sf_elo = 2000
    grade_depth = 14
    threshold = 100

    print(f"playing {n_games} games vs Stockfish@{sf_elo} (300ms/move) to find fresh blunders...",
          flush=True)
    games = play_games(n_games, 300, 300, sf_elo, 1, 200)

    sf = chess.engine.SimpleEngine.popen_uci(DEFAULT_STOCKFISH_PATH)
    sf.configure({"Threads": 1, "Hash": 128})

    candidates = []
    for gi, game in enumerate(games):
        plies = game["plies"]
        we_white = game["we_white"]
        for pi in range(len(plies) - 1):
            fen_before, we_move = plies[pi]
            if not we_move:
                continue
            fen_after = plies[pi + 1][0]
            board = chess.Board(fen_before)
            board_after = chess.Board(fen_after)
            # recover the move we played by diffing legal moves' resulting fens
            played_move = None
            for mv in board.legal_moves:
                board.push(mv)
                if board.board_fen() == board_after.board_fen():
                    played_move = mv
                    board.pop()
                    break
                board.pop()

            info_before = sf.analyse(chess.Board(fen_before), chess.engine.Limit(depth=grade_depth))
            info_after = sf.analyse(chess.Board(fen_after), chess.engine.Limit(depth=grade_depth))
            cp_before = info_before["score"].white().score(mate_score=3000)
            cp_after = info_after["score"].white().score(mate_score=3000)
            adv_before = our_advantage_cp(cp_before, we_white)
            adv_after = our_advantage_cp(cp_after, we_white)
            swing = adv_before - adv_after
            best_move = info_before.get("pv", [None])[0]
            if swing >= threshold:
                candidates.append({
                    "game": gi, "ply": pi, "fen_before": fen_before, "fen_after": fen_after,
                    "we_white": we_white, "played_move": played_move.uci() if played_move else "?",
                    "best_move": best_move.uci() if best_move else "?",
                    "sf_cp_before_white": cp_before, "sf_cp_after_white": cp_after,
                    "swing": swing,
                })

    sf.quit()
    candidates.sort(key=lambda c: -c["swing"])
    print(f"\nfound {len(candidates)} candidate blunders (swing >= {threshold}cp); showing top 3\n",
          flush=True)

    for c in candidates[:3]:
        our_eval_before = our_eval(c["fen_before"])
        our_eval_after = our_eval(c["fen_after"])
        stm = "White" if chess.Board(c["fen_before"]).turn == chess.WHITE else "Black"
        print("=" * 90)
        print(f"Position before OUR move ({stm} to move, we are {'White' if c['we_white'] else 'Black'}):")
        print(f"  FEN: {c['fen_before']}")
        print(f"  Stockfish eval (depth {grade_depth}, White-relative): {c['sf_cp_before_white']}cp")
        print(f"  OUR ENGINE's own static eval (White-relative):        {our_eval_before:.1f}cp")
        print(f"  Stockfish's suggested best move: {c['best_move']}")
        print(f"  Move WE actually played:         {c['played_move']}   <-- the blunder")
        print(f"  Stockfish eval after our move (White-relative): {c['sf_cp_after_white']}cp")
        print(f"  OUR ENGINE's own static eval of the resulting position: {our_eval_after:.1f}cp")
        print(f"  Swing against us (our-side-relative): {c['swing']}cp")
        print()


if __name__ == "__main__":
    main()
