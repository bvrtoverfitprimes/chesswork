"""Extract WDL-labeled quiet positions from our stored game trails for Texel-style
tuning of the classical eval's category scales. Output: fen<TAB>white_result per line."""
import glob
import json
import os
import sys

import chess

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    files = [
        "training/data/raw/rigorous_benchmark_games.jsonl",
        "training/data/raw/overnight_games.jsonl",
    ]
    out = os.path.join(REPO, "training", "tune", "tuning_positions.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n = 0
    seen = set()
    with open(out, "w") as of:
        for rel in files:
            path = os.path.join(REPO, rel)
            if not os.path.exists(path):
                continue
            for line in open(path):
                line = line.strip()
                if not line:
                    continue
                g = json.loads(line)
                we_white = g["we_white"]
                res = g["result"]  # our-POV
                white_res = res if we_white else 1.0 - res
                plies = g["plies"]
                for i, entry in enumerate(plies):
                    if i < 8 or i % 4:  # skip opening, sample every 4th
                        continue
                    fen = entry[0]
                    if fen in seen:
                        continue
                    try:
                        b = chess.Board(fen)
                    except Exception:
                        continue
                    if b.is_check():
                        continue
                    seen.add(fen)
                    of.write(f"{fen}\t{white_res}\n")
                    n += 1
    print(f"wrote {n} WDL-labeled quiet positions to {out}")


if __name__ == "__main__":
    main()
