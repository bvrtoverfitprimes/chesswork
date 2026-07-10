"""Play N recorded games vs calibrated Stockfish, balanced colors, saving EVERY
move of every game (SAN + UCI + FEN trail) plus the result, to a jsonl and a
human-readable PGN-ish log. Default: 10 games, 5 each color, vs SF@2400.

  python tools/record_match.py --sf-elo 2400 --games 10 --ms 1000
"""
import argparse
import json
import os

import chess
import chess.engine
import chess.pgn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(REPO, "tools", "uci_engine_raw.exe")
SF = os.path.join(REPO, "tools", "external", "stockfish", "stockfish",
                   "stockfish-windows-x86-64-avx2.exe")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sf-elo", type=int, default=2400)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--ms", type=int, default=1000)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--binary", default=RAW)
    p.add_argument("--raw", action="store_true", default=True, help="RAW_WEIGHT=1 (classical eval)")
    p.add_argument("--out", default="tools/elo_logs/record_2400_games.jsonl")
    p.add_argument("--pgn", default="tools/elo_logs/record_2400_games.pgn")
    args = p.parse_args()

    env = dict(os.environ)
    if args.raw:
        env["RAW_WEIGHT"] = "1"
    our = chess.engine.SimpleEngine.popen_uci(args.binary, env=env)
    sf = chess.engine.SimpleEngine.popen_uci(SF)
    sf.configure({"UCI_LimitStrength": True, "UCI_Elo": args.sf_elo, "Threads": 1})
    lim = chess.engine.Limit(time=args.ms / 1000)

    score = 0.0
    w = d = l = 0
    outf = open(args.out, "w")
    pgnf = open(args.pgn, "w")
    for g in range(args.games):
        we_white = (g % 2 == 0)  # 5 white, 5 black for even games
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"raw vs SF@{args.sf_elo}"
        game.headers["White"] = "raw" if we_white else f"SF{args.sf_elo}"
        game.headers["Black"] = f"SF{args.sf_elo}" if we_white else "raw"
        node = game
        plies = []
        while not board.is_game_over(claim_draw=True) and board.ply() < args.max_plies:
            we_move = (board.turn == chess.WHITE) == we_white
            fen = board.fen()
            eng = our if we_move else sf
            mv = eng.play(board, lim, game=object()).move
            if mv is None:
                break
            plies.append({"fen": fen, "we_move": we_move, "uci": mv.uci(),
                          "san": board.san(mv)})
            node = node.add_variation(mv)
            board.push(mv)
        res = board.result(claim_draw=True)
        game.headers["Result"] = res
        our_pov = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == we_white else 0.0)
        score += our_pov
        if our_pov == 1.0:
            w += 1
        elif our_pov == 0.5:
            d += 1
        else:
            l += 1
        rec = {"game": g + 1, "we_white": we_white, "result": our_pov,
               "sf_result": res, "plies": plies, "final_fen": board.fen()}
        outf.write(json.dumps(rec) + "\n")
        outf.flush()
        pgnf.write(str(game) + "\n\n")
        pgnf.flush()
        print(f"game {g+1}/{args.games} ({'White' if we_white else 'Black'}): "
              f"{our_pov} [{res}]  running={score}/{g+1}  ({w}W/{d}D/{l}L)", flush=True)

    our.quit()
    sf.quit()
    outf.close()
    pgnf.close()
    pct = 100.0 * score / args.games
    print(f"\nRESULT raw {score}/{args.games} ({pct:.1f}%) vs SF@{args.sf_elo} @ {args.ms}ms "
          f"[{w}W/{d}D/{l}L] -- moves saved to {args.out} and {args.pgn}", flush=True)


if __name__ == "__main__":
    main()
