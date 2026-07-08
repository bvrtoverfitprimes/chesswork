import argparse
import io
import json
import os
import urllib.request

import zstandard

LICHESS_EVAL_URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def stream_filtered_positions(url, target_count, min_depth):
    req = urllib.request.Request(url, headers={"User-Agent": "chesswork-data-pull/1.0"})
    with urllib.request.urlopen(req) as resp:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(resp) as stream_reader:
            text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
            count = 0
            for line in text_stream:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                evals = obj.get("evals", [])
                if not evals:
                    continue
                best_eval = max(evals, key=lambda e: e.get("depth", 0))
                if best_eval.get("depth", 0) < min_depth:
                    continue
                pvs = best_eval.get("pvs", [])
                if not pvs:
                    continue
                top_pv = pvs[0]
                if "cp" not in top_pv and "mate" not in top_pv:
                    continue
                yield {
                    "fen": obj["fen"],
                    "cp": top_pv.get("cp"),
                    "mate": top_pv.get("mate"),
                    "depth": best_eval.get("depth", 0),
                }
                count += 1
                if count >= target_count:
                    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--min-depth", type=int, default=12)
    parser.add_argument("--out", type=str, default="training/data/raw/lichess_sample.jsonl")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    with open(args.out, "w") as f:
        for pos in stream_filtered_positions(LICHESS_EVAL_URL, args.count, args.min_depth):
            f.write(json.dumps(pos) + "\n")
            written += 1
            if written % 500 == 0:
                print(f"{written}/{args.count}", flush=True)

    print(f"wrote {written} positions to {args.out}")


if __name__ == "__main__":
    main()
