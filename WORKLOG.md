# Work Log

## 1. Reproduced chess, then ported to C++

We reproduced chess without any external imports.

Ported the engine to C++ (pieces.h/.cpp, board.h/.cpp) with a two-player
terminal demo. Move input/output follows UCI long algebraic notation,
including promotion suffixes (e.g. e7e8q) and full underpromotion support.

Added FEN-based position setup so arbitrary positions can be constructed
directly, and implemented the full set of game-ending conditions:
checkmate, stalemate, threefold repetition, the fifty-move rule, and
insufficient material (kings only, or king+single minor vs king).

Verified correctness with a test suite (tests/test_chess.cpp) covering:
- perft move counts against known-correct reference values (standard
  start position depths 1-5, matching exactly including promotions, and
  the Kiwipete stress position for castling/pins/en passant)
- castling legality (through check, into check, while in check)
- en passant capture
- all four promotion choices
- pinned pieces being denied illegal moves
- fool's mate checkmate detection
- a hand-built stalemate position
- fifty-move rule and threefold repetition triggers
- insufficient material detection
- an illegal-move strain test trying every from/to square pair against
  the legal move list

Learned: the original Python prototype tracked a halfmove clock but never
used it to end the game, and only ever auto-promoted to queen. Both were
gaps against full chess rules that the test suite surfaced.

## 2. Research: how modern chess engines actually think

Survey of the field, from classical brute-force search to today's neural
approaches, so we know what we're building toward before writing any
engine code.

### 2.1 Board representation: bitboards

Fast engines don't loop over an 8x8 array of chars like our `board_`.
They represent each piece type/color as a 64-bit integer (`bitboard`),
one bit per square. Move generation becomes bitwise AND/OR/XOR/shift
instead of nested loops, which is why it can run millions of times a
second. Sliding pieces (bishop/rook/queen) are the hard part — naively
walking each ray per move is too slow, so engines precompute attack
tables and use "magic bitboards": mask the relevant occupancy bits,
multiply by a found "magic number" to hash into a lookup table of
precomputed attack sets. Roughly a 30% speedup over the classical
ray-walking approach, and it's the standard technique in essentially
every serious engine (Stockfish included).

### 2.2 Classical search: minimax + alpha-beta and its enhancements

The foundational algorithm is minimax over the game tree; alpha-beta
pruning cuts branches that can't affect the result, in the best case
roughly squaring the number of positions searchable per unit of time.
Alpha-beta alone is not enough — the techniques that turn it into a
competitive engine (all still used inside Stockfish today, layered on
top of NNUE evaluation) are:

- **Move ordering** — searching the most promising moves first makes
  alpha-beta cutoffs happen sooner. Good ordering can be a 5-10x speedup
  over random ordering. Built from MVV-LVA (captures ordered by
  victim value minus attacker value), **killer moves** (quiet moves that
  caused a beta cutoff at the same ply in a sibling branch, tried early
  next time), and the **history heuristic** (a table scoring moves by
  how often they've caused cutoffs anywhere, generalizing killer moves).
- **Transposition tables** — a hash table keyed by a Zobrist hash of the
  position, storing previously-searched results so transposing move
  orders that reach the same position aren't re-searched. (Our engine
  already computes a Zobrist hash for repetition detection — the same
  hash is the key input a transposition table would need.)
- **Iterative deepening** — search depth 1, then 2, then 3, etc.,
  rather than jumping straight to the target depth. Looks wasteful but
  the shallow searches feed move-ordering and transposition-table data
  that make the deeper search much faster, and it gives an any-time
  result if a time limit cuts the search off.
- **Quiescence search** — never stop the search in the middle of a
  capture sequence, since the static evaluation of a position mid-trade
  is meaningless. At the search horizon, keep searching captures (and
  usually checks) only, until the position is "quiet."
- **Null move pruning** — assume a side could pass (make a "null move")
  and still comes out ahead; if so, the position is almost certainly
  good enough to prune without a full-depth search. Very powerful,
  usually the first pruning technique added after alpha-beta itself.
- **Late move reductions (LMR)** — moves ordered late (i.e. that move
  ordering thinks are unlikely to be best) are searched to a shallower
  depth first, and only re-searched at full depth if they unexpectedly
  turn out well.
- **Aspiration windows** — start the alpha-beta window narrow, centered
  on the previous iteration's score, and only widen it if the search
  falls outside that window. Narrower windows prune more.
- **Singular extensions** — if one move is far better than every
  alternative at a node, extend the search by a ply specifically on
  that move, since forced sequences deserve deeper looks.
- **Lazy SMP** — the parallel-search approach modern engines use for
  multi-core scaling: every thread searches the whole tree independently
  with almost no cross-thread communication except a shared
  transposition table, which naturally avoids duplicated work as
  threads' search orders diverge and converge.

Handcrafted evaluation (material values, piece-square tables, king
safety, pawn structure, mobility, etc.) used to sit at the leaves of
this search. Stockfish carried a handcrafted eval for years; it was
fully removed in Stockfish 16 (2023) in favor of NNUE.

### 2.3 NNUE — the technique that replaced handcrafted evaluation

NNUE ("Efficiently Updatable Neural Network") was invented in 2018 by
Yu Nasu for computer shogi, then ported into Stockfish in 2019-2020 for
an ~80 Elo jump, and fully replaced Stockfish's handcrafted evaluation
by version 16. It's a small neural network evaluated at every leaf node
of a classical alpha-beta search — NNUE is a *replacement for the
evaluation function*, not a replacement for search.

Key design ideas, all in service of making the network cheap enough to
call millions of times per second on a CPU:

- **Sparse, incremental inputs.** The input layer has one feature per
  (own king square, piece type, piece color, piece square) combination
  — HalfKP originally, now variants like HalfKAv2 with tens of
  thousands of possible features, of which at most ~32 are ever active
  (one per piece on the board). Because moving one piece only changes
  a couple of active input features, the first layer's output (the
  "accumulator") can be *incrementally updated* rather than
  recomputed from scratch after every move — the "efficiently
  updatable" half of the name.
- **Quantized integer inference.** Weights and activations are stored
  as int8/int16 rather than floats, and `ClippedReLU` clips activations
  into a small range (0-127) specifically so the next layer's matrix
  multiply stays in int8 range. This trades a little precision for
  much faster inference on ordinary CPUs (no GPU required).
- **Training target.** The network is trained via gradient descent on
  hundreds of millions of quiescent (no pending captures/mate-in-1)
  positions labeled with position-score pairs, historically Stockfish's
  own search evaluations at moderate depth (self-improving over
  successive network generations) — so NNUE can be thought of as a
  learned "search depth multiplier" for the classical evaluation, not a
  fundamentally different way of judging positions.

### 2.4 AlphaZero / MCTS — the alternative paradigm

DeepMind's AlphaZero (2017) replaced both handcrafted evaluation *and*
alpha-beta search with a completely different pairing: a deep residual
CNN (a "body" of 3x3-kernel residual blocks, model sizes commonly
described as WxD e.g. 128 filters x 10 blocks) with two output heads —
a **policy head** (probability distribution over candidate moves) and
a **value head** (a single scalar estimate of win probability) — driving
**Monte Carlo Tree Search**, specifically **PUCT** (Predictor + Upper
Confidence bound applied to Trees): the tree is expanded by repeatedly
picking the child that balances high policy-network prior with high
value-network estimate against how little it's been explored, rather
than exhaustively expanding every branch like alpha-beta does.

Training is pure self-play reinforcement learning with **zero human
game data**: self-play games (network + MCTS vs. itself) generate
training examples where the policy head is trained to reproduce the
*visit-count distribution* MCTS ended up with (search "amplifies" the
raw policy into a better one, and the network is trained to imitate its
own amplified output), and the value head is trained to predict the
actual game result. AlphaZero reportedly exceeded Stockfish's strength
after only ~4 hours of self-play training (300k steps) starting from
literally nothing but the rules.

**Leela Chess Zero (Lc0)** is the open-source community reproduction of
this approach for chess. It added a Squeeze-and-Excite extension to the
residual blocks, and in February 2024 moved its strongest network
generation (BT4) from convolutional to **transformer** architecture (15
encoder layers, hidden size 1024, 32 attention heads, treating the 64
squares as a token sequence) — a further ~270 Elo policy-accuracy gain
over the prior conv net.

### 2.5 Where things stand right now (mid-2026)

- **Stockfish is the strongest engine in the world**, full stop — an
  estimated ~3653 Elo, and it has won every TCEC Superfinal since Season
  18. It is the classical-search paradigm (alpha-beta + all the
  enhancements in 2.2) with NNUE as its evaluation function — i.e. the
  "old" and "new" paradigms aren't actually in competition inside the
  strongest engine, they're stacked.
- **Leela Chess Zero is a close second**, typically losing head-to-head
  matches by only 20-50 Elo when given comparable hardware, and playing
  a noticeably different, more "positional/prophylactic" style since it
  doesn't rely on raw search depth the way Stockfish does. Lc0 needs a
  capable GPU to be competitive; Stockfish is CPU-only and cheaper to
  run well.
- **Pure transformer / language-model-style chess engines without any
  search at inference time are a very active 2024-2026 research
  direction, but not yet at the top.** Approaches include distilling
  Stockfish's action-value function directly into a causal transformer,
  and decoder-only "chess as language modeling" systems (e.g. Allie).
  These can reach strong (grandmaster-level) play from the policy/value
  network alone with no search, which is a meaningfully different
  engineering tradeoff (cheap single forward-pass inference vs. a
  search loop), but they're a research frontier, not yet displacing
  Stockfish or Lc0 in top play.
- **Practical takeaway for "absolute best right now":** if the goal is
  raw playing strength, replicate Stockfish's recipe — bitboards, deep
  alpha-beta with the full suite of pruning/ordering/parallelization
  tricks, and NNUE as the leaf evaluator. If the goal is a different
  kind of intelligence (more human-like, less brute-force, more
  research-interesting), the AlphaZero/Lc0 self-play + MCTS + deep net
  recipe is the other proven path, and pure-policy transformer models
  (no search at all) are the frontier worth watching but not yet
  state-of-the-art.

Sources consulted: chessprogramming.org (Alpha-Beta, NNUE, Null Move
Pruning, Killer Heuristic, Lazy SMP, Bitboards, AlphaZero, Leela Chess
Zero pages), Stockfish NNUE docs and nnue-pytorch documentation
(official-stockfish/Stockfish and nnue-pytorch on GitHub), Wikipedia
(Efficiently Updatable Neural Network, Alpha-beta pruning), the AlphaZero
paper (Silver et al., Science 2018 / arXiv:1712.01815), Leela Chess Zero
project blog and docs (lczero.org), and 2026 engine-comparison writeups
(checkmatex.app, raindropchess.com) for current Elo/TCEC standings.

## 3. Built a first engine (MVP, no neural network, no learning)

Added an `engine/` folder implementing the classical recipe from section
2.2 at MVP scope: material-count + center-bonus evaluation
(`engine/evaluation.h/.cpp`, the same heuristic the very first Python
prototype used), and negamax with alpha-beta pruning plus simple
capture-first move ordering and mate-distance scoring
(`engine/search.h/.cpp`). Fixed search depth (4 plies), no quiescence
search, no transposition table — deliberately left out to keep this
first version simple; a real strength upgrade would add those next.

Two driver programs consume the same `chess::Game` and `engine::search`
API:
- `engine/engine_gameplay.cpp` — asks which color you want to play, then
  runs a human-vs-engine game in the terminal (engine moves are chosen
  by `findBestMove` and printed before the board).
- `engine/engine_selfplay.cpp` — engine vs. itself for one full game,
  emitting a proper PGN transcript (SAN notation with disambiguation,
  captures, castling, promotion, and check/checkmate symbols) to stdout.

Added `Game::boardArray()` (a const accessor to the private board state)
to `chess/board.h` so the evaluator can read the position without the
engine needing to duplicate board representation.

Verified: `make test` still passes unchanged after adding the accessor;
ran `engine_selfplay` end-to-end (a full game resolved by repetition
draw in ~10s at depth 4) and `engine_gameplay` as both colors, confirming
legal engine moves are chosen and applied correctly in both directions.

## 4. Reorganized entry points, stripped comments

Moved `simple_demo.cpp` into `tests/` (it's a manual smoke test, not a
deliverable). Pulled `engine_gameplay.cpp` and `engine_selfplay.cpp` out
of `engine/` up to the repo root, since they're the two things meant to
actually be run directly — `engine/` now holds only the reusable
evaluation/search library code. Updated all `#include` paths and the
Makefile accordingly, rebuilt, and reran `make test` plus both
root-level programs to confirm nothing broke in the move.

Removed all comments from every source file, including the trailing
`// namespace x` markers on closing braces.

## 5. CURRENT GAME PLAYED

The engine is fully deterministic (fixed-depth alpha-beta, no
randomness), so `engine_selfplay` always produces this same game right
now. Recorded here as a snapshot of current playing strength; this
section should be replaced whenever the engine changes enough to
produce a different game.

```
[Event "Self-play"]
[Site "?"]
[Date "2026.7.6"]
[Round "1"]
[White "SimpleEngine"]
[Black "SimpleEngine"]
[Result "1/2-1/2"]

1. e4 Nc6 2. Bb5 Nd4 3. Bc4 b5 4. c3 bxc4 5. cxd4 c5 6. dxc5 e5 7. b3
Bxc5 8. a3 Bd4 9. Ra2 Qh4 10. d3 Bb6 11. g3 Qf6 12. d4 cxb3 13. Qxb3
Ba5+ 14. Bd2 Bxd2+ 15. Rxd2 Qc6 16. Qc2 exd4 17. Qxc6 dxc6 18. Rxd4 f5
19. e5 Rb8 20. Rb4 Rxb4 21. axb4 Bd7 22. Nf3 Ne7 23. b5 cxb5 24. Nd4
Nc6 25. Nxc6 Bxc6 26. O-O b4 27. f4 Ke7 28. h3 Rg8 29. h4 Rh8 30. Nd2
Rd8 31. Rc1 Rxd2 32. Rxc6 Rd1+ 33. Kg2 Rd2+ 34. Kg1 Rd1+ 35. Kg2 Ke8
36. Rc5 Kf8 37. Rc8+ Kf7 38. Rc7+ Kf8 39. Rxa7 Kg8 40. Ra8+ Kf7 41.
Ra7+ Kf8 42. Ra6 Rd2+ 43. Kg1 Rd1+ 44. Kg2 Rd2+ 45. Kg1 Rd1+ 46. Kg2
1/2-1/2
```

Ends in a draw by repetition around move 46: after trading down to
rook + bishop/knight + pawns each side, White's rook shuffles Black's
rook into a repeated check sequence (`Rd2+`/`Rd1+` vs. `Kg2`/`Kg1`)
that neither side can break out of at this search depth.
