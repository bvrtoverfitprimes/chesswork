# human_limit

The current engine (namespace `human_limit`). Evaluation is a from-scratch NNUE-style
sparse network, trained externally in Python and loaded as a binary weights file; search
is the same classical alpha-beta skeleton used in `old_engine`.

This replaces the previous generation's `0.3*classicalEval + 0.7*networkEval` blend — see
git history / WORKLOG.md for that design and why it existed. This generation's network is
trained on dense externally-labeled regression targets (not sparse self-play game outcomes),
so it passes basic material-recognition sanity checks without needing a classical-eval floor
to fall back on.

## Evaluation: HalfKP-mirrored sparse network

Full encoding spec: `training/encoding/feature_spec.md` (normative; the Python reference
implementation is `training/encoding/encode.py`, the C++ implementation is
`nnue_features.h/.cpp`, and the two are kept in exact agreement by
`tests/test_nnue_features.cpp` against a checked-in golden fixture — this is a hard build
gate, not just documentation).

Summary: each position is seen from two perspectives (side-to-move, other-side). Each
perspective's own king square (after a per-color rank flip and a king-file mirror that
halves the king-bucket space from 64 to 32) selects one of 32 "king buckets"; every other
piece on the board contributes one active feature indexed by
`kingBucket*640 + pieceTypeIdx*64 + squareIdx`. A single shared `20480 x H` embedding table
is summed over each perspective's active features (`H=128` for this generation), the two
resulting vectors are concatenated, and a small dense head
(`Linear(2H,32)->ReLU->Linear(32,32)->ReLU->Linear(32,1)`) produces the final score. The
per-perspective king-file mirroring is simultaneously free data augmentation (a position and
its horizontal mirror share the same weight rows).

`Network::evaluate()` converts the network's side-to-move-relative raw output into the
White-relative score the rest of the codebase (search, `old_engine` comparison) expects —
see the sign-handling comment analog in `network.cpp`.

## Search

Unchanged from `old_engine`: negamax with alpha-beta pruning, transposition table,
iterative deepening under a time budget, quiescence search, null-move pruning, late move
reductions, aspiration windows, layered move ordering. Only the leaf evaluation source
changed.

## Training (moved to Python)

Training no longer happens in C++ (`train_human_limit.cpp` is retired — kept in the repo as
a historical record of the self-play approach used by the previous generation, but it no
longer compiles against the current `Network` API, which dropped `trainStep`/`randomInit`
entirely). All training now lives under `training/`:

- `training/data/download_lichess_subset.py` streams positions directly from the public
  Lichess evaluations database (`database.lichess.org`) without ever writing the full
  (20+ GB compressed) file to disk — it decompresses and filters in-flight and stops the
  connection once enough labeled positions are collected.
- `training/data/build_dataset.py` converts a labeled JSONL file into feature-index tensors
  (`training/data/processed/*.npz`), computing a side-to-move-relative regression target
  from Lichess's White-relative `cp`/`mate` fields (clamped to +-3000cp, scaled by /400).
- `training/model/{net,train,export}.py` defines the PyTorch model, runs supervised
  regression (Huber loss) with Adam, and exports the trained weights to the flat binary
  format `engine/human_limit/nnue_weights.bin` that `Network::load()` reads.
- `training/eval/correlation_eval.py` independently checks the compiled C++ engine's static
  eval against held-out labeled positions (via `tools/eval_cli`), rather than trusting the
  Python training log alone.
- `tools/external/stockfish/` (gitignored) holds a locally-downloaded Stockfish binary, used
  as a teacher to label synthetic/curated positions (endgames, tactics, material-imbalance
  drills) beyond what the Lichess dataset covers by itself — see `training/data/*` as this
  grows in later generations.

Weights are stored in `engine/human_limit/nnue_weights.bin` (gitignored, a training
artifact, not source code) — a small binary header (`magic`, `hidden` size) followed by the
embedding table and head weights in a fixed order `export.py` and `network.cpp` both agree
on.

## Known limitations of this generation, stated rather than hidden

- Trained on a modest slice of the Lichess evaluations dataset (a few hundred thousand
  positions) — enough to clear a first correctness/sanity bar, not yet the several-million-
  position scale a stronger generation should use.
- Inference recomputes both perspectives' accumulators from scratch on every node (no
  incremental accumulator yet) — search speed is currently unchanged from `old_engine`.
- `play.html`'s in-browser JS port and `tools/weights_to_js.py` still reflect the *previous*
  (20-hand-feature) architecture and were deliberately left as a frozen historical snapshot
  rather than ported to the new architecture, by explicit decision — same treatment already
  given to `ancient_engine`/`old_engine`.
