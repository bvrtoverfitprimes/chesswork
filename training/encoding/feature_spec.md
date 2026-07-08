# NNUE feature encoding (HalfKP-mirrored)

Normative spec. `encode.py` (Python) and `engine/human_limit/nnue_features.h/.cpp` (C++)
are two independent implementations of exactly this arithmetic; `gen_golden.py` /
`golden_features.json` / `tests/test_nnue_features.cpp` are the gate that proves they agree.

Board convention: row 0 = rank 8, row 7 = rank 1, col 0 = file a, col 7 = file h
(matches `chess::BoardArray` and FEN reading order).

## Per-perspective transform

Two perspectives are computed per position: `stm` (side to move) and `ntm` (the other side).

For perspective color `persp`:

1. **Rank flip by color**: if `persp == Black`, every square `(r, c)` used below is first
   transformed to `(7 - r, c)`. If `persp == White`, squares are used as-is. (This is a rank
   flip only, not a full 180 rotation — files are untouched here.)
2. **King-file mirror**: find `persp`'s own king's square after step 1. If its column `>= 4`
   (files e-h), mirror every square's column `c -> 7 - c` for this perspective only. This is
   what halves the king-bucket space from 64 to 32, and is simultaneously free data
   augmentation (a position and its horizontal mirror share the same weight rows).
3. **King bucket**: after steps 1-2, the own king sits at `(kr, kc)` with `kc in 0..3`.
   `kingBucket = kr * 4 + kc`, range `0..31`.
4. **Per-piece feature**: for every non-king piece remaining on the board, after applying the
   same steps 1-2 transform to its square `(r, c)`:
   - `pieceTypeIdx`: own pawn/knight/bishop/rook/queen = `0..4`; opponent
     pawn/knight/bishop/rook/queen = `5..9` (own/opponent relative to `persp`, not to White).
   - `sqIdx = r * 8 + c` (post-transform), range `0..63`.
   - `featureIndex = kingBucket * 640 + pieceTypeIdx * 64 + sqIdx`, range `0..20479`
     (`640 = 10 piece-type slots * 64 squares`, `32 buckets * 640 = 20480` total).

Kings themselves are never features (they are the anchor, not an input). Castling rights and
en passant are deliberately not encoded — out of scope for this generation, standard for
HalfKP-family nets.

## Network input

`concat(accumulator(stm_features), accumulator(ntm_features))` where `accumulator` is a
`sum` over one shared `20480 x H` embedding table (both perspectives read/write the same
table — halves parameter count vs. separate tables per perspective).

## Output of `fen_to_feature_indices(fen)`

Two lists of active feature indices (each length = number of non-king pieces on the board,
at most 30): `(stm_indices, ntm_indices)`.
