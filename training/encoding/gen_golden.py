import json
import os

from encode import fen_to_feature_indices, fen_to_output_bucket

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "r3k2r/ppp2ppp/2n1bn2/2bqp3/2B1P3/2NP1N2/PPP1QPPP/R1B2RK1 b kq - 2 9",
    "1r3rk1/ppp2ppp/2n1bn2/2bqp3/2B1P3/2NP1N2/PPP1QPPP/2KR3R w - - 4 12",
    "5rk1/ppp2ppp/2n1bn2/2bqp3/2B1P3/2NP1N2/PPP1QPPP/2KR3R w - - 4 12",
    "6k1/ppp2ppp/2n1bn2/2bqp3/2B1P3/2NP1N2/PPP1QPPP/2KR3R w - - 4 12",
    "8/8/8/8/8/8/4Q3/4K1k1 w - - 0 1",
    "8/8/8/8/8/8/2Q5/1K4k1 w - - 0 1",
    "8/8/8/8/8/8/8/R3K2k w - - 0 1",
    "8/8/8/8/8/8/8/4KR1k w - - 0 1",
    "8/8/8/8/8/8/4P3/4K1k1 w - - 0 1",
    "8/8/8/8/8/8/8/4K1kq w - - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1",
    "6k1/8/8/8/8/8/8/QQQQKQQQ w - - 0 1",
    "r1bq1rk1/ppp2ppp/2n1bn2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1",
    "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5P2/PPPP2PP/RNBQKBNR w KQkq - 2 3",
    "8/8/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/8/k6K b - - 0 1",
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 2 2",
    "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/1P6/8/8/8/8/6p1/K6k w - - 0 1",
    "3q4/8/8/8/8/8/8/K6k w - - 0 1",
    "8/8/8/8/8/8/8/1K5k b - - 0 1",
]

FILE_LETTERS = "abcdefgh"
for f in range(8):
    white_sq = FILE_LETTERS[f] + "1"
    black_sq = FILE_LETTERS[7 - f] + "8"
    placement = ["8"] * 8
    board_rows = [[" "] * 8 for _ in range(8)]
    board_rows[7][f] = "K"
    board_rows[0][7 - f] = "k"
    board_rows[7][(f + 3) % 8] = "R" if (f + 3) % 8 != f else "Q"
    board_rows[0][(7 - f + 2) % 8] = "n" if (7 - f + 2) % 8 != (7 - f) else "b"
    rows = []
    for row in board_rows:
        s = ""
        empty = 0
        for ch in row:
            if ch == " ":
                empty += 1
            else:
                if empty:
                    s += str(empty)
                    empty = 0
                s += ch
        if empty:
            s += str(empty)
        rows.append(s)
    fen_w = "/".join(rows) + " w - - 0 1"
    fen_b = "/".join(rows) + " b - - 0 1"
    FENS.append(fen_w)
    FENS.append(fen_b)


def main():
    golden = []
    for fen in FENS:
        stm, ntm = fen_to_feature_indices(fen)
        bucket = fen_to_output_bucket(fen)
        golden.append({"fen": fen, "stm": sorted(stm), "ntm": sorted(ntm), "bucket": bucket})

    json_path = os.path.join(os.path.dirname(__file__), "golden_features.json")
    with open(json_path, "w") as f:
        json.dump(golden, f, indent=1)

    txt_path = os.path.join(os.path.dirname(__file__), "golden_features.txt")
    with open(txt_path, "w") as f:
        for g in golden:
            f.write(g["fen"] + "|" + ",".join(map(str, g["stm"])) + "|" + ",".join(map(str, g["ntm"])) +
                     "|" + str(g["bucket"]) + "\n")

    print(f"wrote {len(golden)} golden fixtures to {json_path} and {txt_path}")


if __name__ == "__main__":
    main()
