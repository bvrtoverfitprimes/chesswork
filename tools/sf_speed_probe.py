import os
import sys

import chess
import chess.engine

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SF_PATH = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                        "stockfish-windows-x86-64-avx2.exe")


def probe(fen, time_s):
    board = chess.Board(fen)
    sf = chess.engine.SimpleEngine.popen_uci(SF_PATH)
    sf.configure({"Threads": 1})
    info = sf.analyse(board, chess.engine.Limit(time=time_s))
    sf.quit()
    nodes = info.get("nodes", 0)
    depth = info.get("depth", 0)
    nps = info.get("nps", 0)
    score = info.get("score")
    print(f"stockfish:   move={info.get('pv', [None])[0]} depth={depth} nodes={nodes} "
          f"score={score} nps={nps}")


if __name__ == "__main__":
    fen = sys.argv[1]
    time_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    probe(fen, time_s)
