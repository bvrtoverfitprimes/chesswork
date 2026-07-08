import subprocess
import sys
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

ELOS = [1500, 1800, 2100, 2400, 2700, 3000]
GAMES_PER_ELO = 8

for elo in ELOS:
    print(f"\n=== Stockfish UCI_Elo={elo} ===", flush=True)
    subprocess.run([PY, os.path.join(REPO, "tools", "match_vs_stockfish.py"),
                     "--engine", "human", "--elo", str(elo),
                     "--games", str(GAMES_PER_ELO),
                     "--our-time-ms", "500", "--sf-time-ms", "500"], check=False)
