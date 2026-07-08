import os
import subprocess
import sys
import itertools
import chess

ENGINES = ["ancient", "old", "human"]
GAMES_PER_PAIRING = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TIME_MS = int(sys.argv[2]) if len(sys.argv) > 2 else 300
MAX_PLIES = 200

CLI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "bestmove_cli.exe")


def get_move(engine, board):
    fen = board.fen()
    out = subprocess.run([CLI, engine, fen, str(TIME_MS)], capture_output=True, text=True, timeout=30)
    uci = out.stdout.strip()
    return chess.Move.from_uci(uci)


def play_game(white_engine, black_engine):
    board = chess.Board()
    for _ in range(MAX_PLIES):
        if board.is_game_over(claim_draw=True):
            break
        engine = white_engine if board.turn == chess.WHITE else black_engine
        move = get_move(engine, board)
        board.push(move)
    result = board.result(claim_draw=True)
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    return 0.5


def main():
    scores = {(a, b): [0.0, 0] for a in ENGINES for b in ENGINES if a != b}

    for a, b in itertools.combinations(ENGINES, 2):
        for g in range(GAMES_PER_PAIRING):
            if g % 2 == 0:
                white, black = a, b
                result_white = play_game(white, black)
            else:
                white, black = b, a
                result_white = play_game(white, black)

            score_a = result_white if white == a else (1.0 - result_white)
            scores[(a, b)][0] += score_a
            scores[(a, b)][1] += 1
            scores[(b, a)][0] += (1.0 - score_a)
            scores[(b, a)][1] += 1

            print(f"{white} vs {black}: white_result={result_white}", flush=True)

    ratings = {e: 1500.0 for e in ENGINES}
    ratings["old"] = 1500.0

    for _ in range(2000):
        for a, b in itertools.combinations(ENGINES, 2):
            total_a, count_a = scores[(a, b)]
            if count_a == 0:
                continue
            score_frac = total_a / count_a
            expected = 1.0 / (1.0 + 10 ** ((ratings[b] - ratings[a]) / 400.0))
            error = score_frac - expected
            k = 2.0
            delta = k * error * count_a / GAMES_PER_PAIRING
            if a != "old":
                ratings[a] += delta
            if b != "old":
                ratings[b] -= delta
        for e in ENGINES:
            ratings[e] = max(200.0, min(3000.0, ratings[e]))

    print("\n=== Results ===")
    for a, b in itertools.combinations(ENGINES, 2):
        total_a, count_a = scores[(a, b)]
        if count_a == 0:
            continue
        print(f"{a} vs {b}: {total_a}/{count_a} ({100.0 * total_a / count_a:.1f}%)")

    print("\n=== Internal relative ratings (anchored: old_engine = 1500, NOT calibrated to real FIDE/CCRL scale) ===")
    for e in ENGINES:
        print(f"{e}: {ratings[e]:.0f}")


if __name__ == "__main__":
    main()
