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

## 5. CURRENT MODEL DESCRIPTION (at the time of section 4)

The engine at this point was the MVP from section 3: material +
center-bonus evaluation, plain minimax with alpha-beta, no
transposition table, no quiescence search. It was fully deterministic
(fixed-depth alpha-beta, no randomness), so `engine_selfplay` always
produced the same game below every time it was run at that stage.
Recorded here as a snapshot of that model's playing strength.

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

## 6. Reproduced state-of-the-art classical engine techniques

Goal: before attempting anything novel, see how strong we could get by
faithfully reproducing the classical recipe surveyed in section 2 —
not to beat top engines, just to find our own ceiling with this
approach.

Archived the old MVP engine (material + center-bonus eval, plain
minimax/alpha-beta) into `engine/old_engine/` under its own
`old_engine::` namespace; confirmed it still compiles and runs
standalone, kept purely as a historical reference point.

Extended the core rules engine (`chess/board.h/.cpp`) with what a real
search needs: a public Zobrist hash accessor, a proper side-to-move
hash component (fixing a latent gap where two positions differing only
in whose turn it was could hash identically), null-move support
(`makeNullMove`/`unmakeNullMove`), and FEN export (`toFen()`). Full
existing test suite (perft, castling, en passant, promotions,
checkmate/stalemate/draws) still passes unchanged after all of this.

Built a new evaluation (`engine/evaluation.cpp`): tapered piece-square
tables (the well-known "PeSTO"-style midgame/endgame tables) plus
material, interpolated by a computed game phase, replacing the old
flat material+center-bonus heuristic.

Built a new search (`engine/search.cpp`): a `Searcher` class with a
transposition table, iterative deepening under a time budget,
quiescence search (including proper check-evasion handling so it
doesn't misjudge being mated at the search horizon), null-move
pruning, late move reductions, aspiration windows, and layered move
ordering (transposition-table move, then MVV-LVA captures, then killer
moves, then history heuristic).

Verified correctness and play quality directly, without adding any
scoring/rating infrastructure to the repo: confirmed the search
respects its time budget correctly across a range of budgets (100ms to
3000ms); hand-built tactical test positions confirmed the engine finds
forced mates (fool's-mate pattern, a back-rank mate-in-1) and avoids
obvious blunders (takes free hanging material, moves an attacked queen
to safety); ran full self-play games end-to-end with no crashes.

Added `play.html` at the repo root: an in-browser way to play against
the engine, with color selection, all four promotion-piece choices,
and a reset button. This is a from-scratch JavaScript reimplementation
of the same rules logic and evaluation tables, not a compiled copy of
the C++ engine — compiling the real engine to WebAssembly would need
installing a large toolchain (LLVM/clang/Node via emscripten), which
was judged too risky given limited local disk space. Verified the JS
implementation independently: it reproduces the exact same perft
counts as the C++ engine through depth 4, finds fool's mate, and
completes multi-ply self-play games without errors.

The new engine is fully deterministic given a time budget only in the
sense that it always tries to use the full budget, but which moves it
finds within that budget can vary run to run depending on machine load
(the time cutoff can land mid-iteration), so `engine_selfplay` no
longer reliably reproduces the exact same game every run. The game
below is one representative run.

```
[Event "Self-play"]
[Site "?"]
[Date "2026.7.6"]
[Round "1"]
[White "SimpleEngine"]
[Black "SimpleEngine"]
[Result "0-1"]

1. Nf3 d5 2. d4 Nf6 3. e3 g6 4. Nc3 Bg7 5. Bb5+ c6 6. Bd3 Be6 7. O-O
O-O 8. e4 dxe4 9. Nxe4 Nxe4 10. Bxe4 Nd7 11. Ng5 Bd5 12. Bxd5 cxd5
13. Be3 e5 14. dxe5 Nxe5 15. c3 Nc4 16. Qg4 h6 17. Nf3 Nxe3 18. fxe3
Qb6 19. Rf2 Rae8 20. Rd1 Rxe3 21. Rxd5 Rfe8 22. Nd4 Re1+ 23. Rf1 Qxb2
24. Qd7 Rxf1+ 25. Kxf1 Qc1+ 26. Kf2 Qe3+ 27. Kf1 Qe1# 0-1
```

A real, recognizable opening (Nf3/d4/Nc3 into a King's-Indian-style
setup), a genuine tactical middlegame with piece trades starting move
8, and a real forced mating sequence finishing with `Qe1#` — a
substantial step up from section 5's old MVP engine, which only ever
shuffled into a repetition draw.

## 7. Known weakness: converting a won endgame to mate

Ad-hoc testing surfaced a real gap: in a position where the engine had
built an overwhelming material advantage (queen and rook vs. a bare
king), it failed to find the forced mate and instead repeated checks
until the game was drawn by threefold repetition — throwing away a
completely winning position.

This is a known class of problem for engines built the way ours is:
the search and evaluation are tuned for finding good moves in roughly
balanced positions, not for the specific technique of mating with
overwhelming material, and nothing in the search actively steers away
from repeating when its own score is already very high. Worth fixing
before pushing the engine's strength further — likely via a mating-net
heuristic in the evaluation, or by making the search treat repetition
as actively bad (rather than neutral) whenever it is winning by a
large margin.

## 8. Innovation phase: a self-trained neural evaluator ("human_limit")

Goal stated up front: not to beat top engines, but to move past
reproducing known techniques and actually train something ourselves —
targeting a realistic ~3000-level strength ceiling (a bar classical
engines cleared decades ago, e.g. Deep Blue in 1997, without any
learning at all), using the same training *paradigm* AlphaZero/Leela
use (learning purely from self-play, no external game data, no other
engine's evaluations used anywhere) scaled down to a much smaller
network and compute budget.

**Renamed and archived** the engine lineage so every stage stays on
record: the original MVP is now `engine/ancient_engine/` and the
classical hand-tuned engine (section 6) is now `engine/old_engine/`,
each with a README explaining its own approach. `human_limit` is the
new current engine.

**Architecture**: a mixed classical/learned evaluation —
`score = 0.3 * classicalEval + 0.7 * networkEval`. `classicalEval` is
the exact same tapered PST formula as `old_engine` (deterministic,
always structurally correct). `networkEval` is a small feedforward
network (20 engineered input features -> 128 -> 64 -> 32 -> 1 linear
output, ~13,000 weights, all learned not hand-set) over the same kind
of engineered features (material, mobility, king safety, pawn
structure, etc.) rather than a raw sparse per-square encoding — a
deliberate scope decision, since a full sparse NNUE-style input needs
an incrementally-updated accumulator threaded through search to stay
fast, which is real engineering work saved for a future generation.
Search is unchanged from `old_engine` (same TT/quiescence/null-move/
LMR/aspiration-window skeleton) — only the leaf evaluation changed.

**Training** (`train_human_limit.cpp`): self-play games generate
positions labeled with the game's actual result; a persistent replay
buffer accumulates across generations (capped, oldest evicted) rather
than training on and discarding each generation's small batch; plus a
batch of synthetic "obvious material imbalance" positions injected
each generation (random self-play opening, then 1-3 random non-king
pieces removed from one side, labeled by which side now has more
material) to directly teach a pattern self-play rarely produces
organically.

**Two real bugs found and fixed along the way, not glossed over:**

1. Games that hit the ply cap without a decided result were being
   silently discarded from training entirely — most of a generation's
   compute was wasted producing data that never got used. Fixed by
   labeling ply-cap games as a draw and using their samples.
2. A genuine scale-mismatch bug: `Network::evaluate()` scales its
   output by 1000 for external use, but training compared that 1000x
   target against the network's *unscaled* raw output — a 1000x
   mismatch producing huge first-step gradients that corrupted the
   learned weights into responding backwards to material features
   (confirmed directly: the network scored capturing a free rook as
   worse than not capturing it, in both a bare endgame and a dense
   middlegame test position — a genuine value-function bug, verified
   by checking the network's raw evaluation, not just search output).
   Fixed by training entirely in the network's native unscaled range.

**Verification, not just claims**: after each fix, re-ran a fixed set
of hand-built tactical positions (free hanging rook in a bare endgame,
free hanging rook in a dense middlegame with a full army still on the
board, a back-rank mate-in-1, and a hanging-queen-safety check) and
confirmed via direct evaluation calls (not just search output) exactly
what the network was and wasn't learning at each stage, rather than
assuming a fix worked. All four tests pass on the final trained
weights.

**Internal Elo ladder** (`tools/rating_ladder.py` + `tools/bestmove_cli`,
no Stockfish involved anywhere): a round-robin among all three engines,
8 games per pairing, 400ms/move, ratings fit via iterative logistic
regression from the actual game scores and anchored arbitrarily at
`old_engine` = 1500 — this is a **relative-only internal scale**, not
calibrated to CCRL/FIDE/any external rating pool, so it says nothing
about proximity to real-world 3000. Results:

- `ancient_engine` vs `old_engine`: 1.5/8 (18.8%)
- `ancient_engine` vs `human_limit`: 4.0/8 (50.0%)
- `old_engine` vs `human_limit`: 6.0/8 (25.0% for human_limit)
- Fitted ratings: ancient 1274, old 1500 (anchor), human_limit 1290

Honest reading: `human_limit` currently sits roughly level with the
simplest baseline (`ancient_engine`) and clearly behind the fully
hand-tuned classical engine. Since both `old_engine` and `human_limit`
share identical search code, this gap is purely about evaluation
quality — the professionally-established classical PST constants still
outperform a network trained on only ~300 total self-play games at a
shallow 3-ply search depth during data generation. The training isn't
hurting (roughly matching the baseline rather than falling below it),
but it hasn't yet clearly surpassed hand-tuning either. Expected for
this scale of self-play budget, not a sign anything is broken.

**`play.html`** ported to match: same mixed-architecture blend and
20-feature extraction in JavaScript, with the actual trained weights
spliced in (verified independently via Node — same free-rook and
opening-move behavior as the C++ version).

**Not yet done, called out explicitly rather than left implicit**: the
move-generation speed rewrite identified by the research in section 9
below (this section's engine is still exactly as fast/slow as
`old_engine` — no speed work landed here); and the plan going forward,
per direct instruction, is to come back and substantially renovate
`human_limit`'s architecture (network size/features, training budget,
possibly the incremental-accumulator NNUE upgrade) aiming at the 3000
target, rather than treating this generation as final.

## 9. Researched how to make search fast (Stockfish-level depth/speed)

Dispatched a research agent (with our actual code as context, not
generic advice) to find out how engines reach depth 15-20 in a few
seconds on ordinary hardware, and what specifically to change in ours.
Findings, prioritized by expected impact:

1. **Biggest suspected-and-confirmed bottleneck**: `getValidMovesUci()`
   generates pseudo-legal moves then does a full make/unmake +
   check-scan for *every single one* just to filter out illegal moves.
   The standard fix is computing checking/pinned pieces once per
   position up front (a ray-scan from the king in each direction) and
   masking move generation directly, never simulating a move just to
   discover it's illegal. Medium effort, likely the single biggest win.
2. **`std::string`-based moves throughout the hot path** (TT entries,
   killers, history keys, move lists) — allocation and hashing cost on
   every node. Fix: packed-integer move encoding, fixed-capacity move
   lists instead of `std::vector<std::string>`. Low effort, high ROI,
   composes with everything else.
3. **Bitboards + magic bitboards** — the largest single engineering
   item (1-3 weeks), valuable but reportedly less singularly dominant
   than fixing 1 and 2 first; real payoff is that it makes pin/check
   detection and future NNUE feature-diffing cheap bitwise ops instead
   of loops over a char grid.
4. **True incremental NNUE accumulator** (add/subtract a few weight
   rows per move instead of recomputing the whole network from scratch
   at every leaf) — medium-high effort, most valuable once/if a larger
   sparse-feature network is built.
5. Lower-effort extras: TT replacement-policy tuning, avoiding heap
   allocation in hot paths generally, `-O3 -march=native`/LTO,
   SIMD (mostly relevant once bitboards exist).

**Applied now**: `-O3 -march=native -flto` in the Makefile (the free,
zero-risk win) — verified the full test suite still passes unchanged.
**Deliberately not attempted this session**: items 1-4 all touch
`chess/pieces.cpp`/`chess/board.cpp`, the shared core underneath all
three engines and the entire test suite — rewriting that mid-session,
while three engines depend on it and training was actively running,
risked a regression across everything at the worst possible time.
Documented here as a prioritized, ready-to-execute backlog rather than
attempted under time pressure.

## 10. Full NNUE revamp of human_limit: new training pipeline, real regression

Explicit direction from here forward: drop the classical/network blend
entirely, go all-in on a properly-sized learned network, bring in
Stockfish and external data as teachers, and target the real ~3000
territory rather than treating `human_limit` as a finished generation.
Given no GPU on this machine (Intel UHD 630 only, 6 CPU cores), the
target architecture is deliberately Stockfish's own NNUE recipe (sparse
king-relative features + a small incrementally-updatable net trained by
supervised regression), not a literal GPU-scale AlphaZero net —
explained and agreed with direct confirmation before starting.

**New feature encoding**: HalfKP-mirrored, specified once in
`training/encoding/feature_spec.md` and implemented independently in
Python (`training/encoding/encode.py`) and C++
(`engine/human_limit/nnue_features.h/.cpp`) — the two are kept
bit-for-bit in agreement by a checked-in golden fixture
(`training/encoding/golden_features.{json,txt}`, 42 positions covering
every king-file-mirror branch, promotions, and castling-adjacent
placements) verified by `tests/test_nnue_features.cpp`, now part of
`make test`. Per-perspective king-file mirroring halves the king-bucket
space (64→32) and is simultaneously free data augmentation.

**New data pipeline** (`training/data/`): a streaming downloader
(`download_lichess_subset.py`) pulls positions directly from the public
Lichess evaluations database over HTTP, decompressing and filtering
in-flight and stopping the connection once enough positions are
collected — the full dataset is 21+GB compressed and is never written
to disk in full. Confirmed empirically (not assumed) that the dataset's
`cp`/`mate` fields are White-relative regardless of whose move it is
(76.8% sign agreement against simple material diff on ≥900cp
imbalances, well above chance), which matters for building a correct
side-to-move-relative training target. Official Stockfish (sf_18,
AVX2 build) downloaded to `tools/external/stockfish/` (gitignored) as a
teacher for later synthetic/curated positions, not yet used for
training in this session.

**New training pipeline** (`training/model/`, PyTorch, CPU): a shared
`20480 x H` embedding table (`H=128` this generation) summed per
perspective via `EmbeddingBag`, feeding a small
`Linear(2H,32)→ReLU→Linear(32,32)→ReLU→Linear(32,1)` head, trained by
Huber regression directly against clamped/scaled centipawn targets
(dense, not the previous generation's sparse ±1/0 game-result signal).
Trained on 400k streamed Lichess positions (an ~380k/20k train/val
split) in under 2 minutes on CPU. Weights exported to a small flat
binary format (`engine/human_limit/nnue_weights.bin`, gitignored,
replacing the old `weights.txt` convention).

**Honest measurement, not just the training log's headline number**:
the training log reported `val_corr=0.847`, but re-measuring directly
(bypassing the log) showed this is partly inflated by a handful of
high-leverage near-mate positions — full-distribution Pearson r is
closer to 0.75, and ~0.61 with those extremes excluded (MAE ≈142cp on
the non-extreme bulk). Still clears a "this network learned something
real" bar by any of these measures, and passes basic material-sanity
checks the old 20-feature network failed outright (white/black up a
queen or minor pieces, or already holding extra material on the board,
all score with correct sign and large magnitude) — but the honest
number is the qualified one, not the flattering one.

**A real regression, found and root-caused, not glossed over**: dropping
the blend and wiring in the new network initially made `human_limit`
play *worse* than the previous generation (7/8 losses to `old_engine`
at 300ms/move, versus the old blended net's 6/8) despite the network
itself being clearly better by every static measure above. Root cause,
confirmed by direct node-throughput measurement rather than guessed:
the new eval was ~25% more expensive per node than the old blend, and
at these shallow blitz-search depths (5-7 ply) where null-move/LMR
pruning makes node growth nearly linear, that constant-factor slowdown
cost a full 1-2 plies of depth — which matters more than eval quality
in ply-limited alpha-beta search. Two fixes landed in
`engine/human_limit/`, both isolated to this engine only:

1. `Network::forward()` was heap-allocating four `std::vector`s on
   *every single evaluation call* (the old hand-rolled net used stack
   arrays) — fixed with reused scratch buffers sized once at `load()`.
2. The real fix: a genuine incremental NNUE accumulator
   (`engine/human_limit/accumulator.h/.cpp`), maintaining one
   persistent `White`/`Black`-keyed accumulator pair per search node
   instead of recomputing all ~30-60 active features from scratch at
   every leaf. Two per-color accumulators are diffed on every move
   (add/remove a handful of embedding rows for the squares that
   actually changed) and only fully recomputed when that color's own
   king moves (the one case where the king-bucket/mirror can change).
   Threaded through `Searcher` as a ply-indexed stack pushed/popped in
   lockstep with every `makeMove`/`unmakeMove`/null-move in
   `negamax`/`quiescence`/the root loop. **Zero changes to
   `chess/board.{h,cpp}` or `chess/pieces.{h,cpp}`** — the existing
   `UndoMove` struct already carries a complete per-move diff, so the
   accumulator stack lives entirely inside `engine/human_limit/`,
   leaving `ancient_engine`/`old_engine` and the shared test suite
   provably untouched.

**Correctness gate for the accumulator, treated as non-negotiable**:
`tests/test_accumulator.cpp` random-walks six games (including a
castling-rich position, a real en-passant-heavy endgame, and a pawn
promotion race) and asserts the incremental accumulator matches an
independent from-scratch recompute after *every single move* — all
439 checks passed bit-exact (`diff=0.000000`) before this was trusted
in the actual search.

**Net effect, measured, not asserted**: node throughput improved from
~75% to ~91% of `old_engine`'s at the same position, and game strength
against `old_engine` at 300ms/move improved from 1/8 (12.5%) →
1.5/6 (25%, heap-alloc fix only) → 4/10 (40%, with the full incremental
accumulator) via `tools/quick_match.py` (a lighter-weight two-engine
match script added alongside `tools/rating_ladder.py`). Not yet a clear
win over `old_engine`, but a real and substantial recovery, and the
honest state to build on rather than declaring victory early.

**Deliberately not done in this session (at time of writing above)**:
`training/data/` doesn't yet use Stockfish as a teacher for synthetic/
curated positions — Stockfish is downloaded and verified but only
Lichess data has been used for training so far. `play.html`/
`tools/weights_to_js.py` were deliberately left as a frozen historical
snapshot of the previous (20-feature) architecture, by explicit
decision, rather than ported to the new one.

## 11. Move-gen speed (pin/check masks) — done, verified, mixed real effect

Tackled the other backlog item from section 9: `Game::getValidMovesUci()`
previously filtered every pseudo-legal move by actually making it,
checking for check, then unmaking it — the classic slow way to compute
legal moves. Replaced with a pin/check-mask precomputation
(`chess::computeLegalMoveContext`/`isLegalFast` in
`chess/pieces.h/.cpp`): find checkers once, compute per-square pin
lines once, then filter each candidate move in O(1). King moves,
castling, and en-passant captures deliberately stay on the old
make/unmake path — the rare discovered-check-through-en-passant and
king-stepping-back-along-a-check-ray cases are exactly where fast
legal-move generators are notorious for subtle bugs, and the slow path
was already proven correct, so there was no reason to risk it for a
small fraction of moves.

**Verification before trusting it, not after**: kept the original
implementation permanently as `Game::getValidMovesUciSlow()` (documented
as a testing oracle, not dead code) and added
`tests/test_fast_legality.cpp`, which cross-validates the fast and slow
methods against each other as exact move-set comparisons across 3018
positions — 30 random self-play walks plus hand-crafted adversarial
positions (pins on every axis, double check, knight/pawn checks with no
block squares, an en-passant-discovered-check setup, a rook pinned next
to a castling path). Zero mismatches. The full existing perft suite
(`tests/test_chess.cpp`, including the dedicated "pinned knight cannot
move off the pin line" test) also passes unchanged.

**Real speedup, measured**: node throughput roughly doubled for both
engines (`old_engine` ~94k→~244k nodes/sec at a sample position,
`human_limit` ~86k→~139k). But this **is not evenly distributed** —
`old_engine`'s bottleneck was almost entirely move generation (its own
eval is cheap), so removing it let `old_engine` search dramatically
deeper for free. `human_limit`'s per-node cost is now dominated by the
NNUE eval itself (untouched by this change), so its speedup was
smaller. Net effect on `tools/quick_match.py` at 300ms/move: the score
that had climbed to 40% (end of section 10) **dropped back to 5%** —
not a bug, a real and expected consequence of a shared-infrastructure
win that happened to benefit the simpler engine more. Recorded honestly
rather than only reporting the parts of this change that looked good:
raw search speed is now substantially better for the whole codebase
(valuable groundwork for the actual ~3000 target either engine would
eventually need), but it temporarily widened `human_limit`'s gap against
`old_engine` rather than closing it. The lever that actually matters for
`human_limit`'s competitiveness now is eval *quality* per ply, not
search speed — which is why the plan going forward is training-heavy
(scale up data volume, bring in the already-downloaded Stockfish binary
as a teacher for synthetic/curated positions) rather than further speed
work.

## 12. Training-heavy phase: architecture resize, Stockfish teacher, real data scale

Explicit direction: size the architecture appropriately and go training-heavy,
targeting 2500 first, then working toward 3000, with research to inform the
approach rather than guessing.

**Architecture resize** (`training/model/net.py`, `engine/human_limit/network.h/.cpp`):
widened the accumulator from H=128 to H=384, and added Stockfish's real output-bucketing
scheme — 8 output subnetworks selected by `(popcount(all pieces)-1)/4`, confirmed via
research against live Stockfish source rather than guessed (an earlier guess of
non-pawn-material-based bucketing was wrong and corrected). The bucket computation is
cross-validated bit-exact between Python and C++ via the same golden-fixture-test discipline
used for the feature encoding. Training loss switched to WDL-space regression with a 2.6
exponent (both figures confirmed from Stockfish's own `docs/nnue.md`), rather than plain
cp-space Huber — full research writeup in `training/research_notes.md`. Deliberately did
*not* adopt Stockfish's Clipped ReLU activation: its rationale is quantization-driven and we
do plain unquantized double-precision inference, so copying the clamp range would risk
bottlenecking the model for no corresponding benefit — kept plain ReLU, a reasoned deviation
rather than a blind copy.

**Stockfish-teacher curriculum** (`training/data/curriculum_positions.py`,
`stockfish_teacher.py`): generates self-play, post-capture, material-imbalance, and explicit
endgame-skeleton (KR/KQ/KP/KBN/KRR/KRP vs K) positions, labels them via the locally-downloaded
Stockfish binary at depth 14 (deeper than Stockfish's own typical self-play label depth of
8-12, per the research) using multiprocessing across all 6 cores (~44 positions/sec). Produced
122,607 labeled positions targeting exactly the endgame-conversion weakness this project has
tracked since section 7 of this log.

**Scaled data**: pulled 6,000,000 Lichess-eval positions (same streaming, never-touches-disk-
in-full approach as before) and merged with the curriculum set for 6.1M total training
positions.

**A real overfitting incident, caught and fixed, not glossed over**: the first full training
run (10 epochs, `train.py` only saving the final epoch's weights) showed `train_loss`
improving monotonically the entire time while validation metrics (`val_mae`, `val_corr`)
peaked around epoch 4-6 and then got steadily worse — textbook overfitting. Because `train.py`
only saved the last epoch, the saved checkpoint was the single worst-performing epoch in the
whole 10-epoch run. Fixed by adding best-checkpoint tracking (save whenever `val_mae`
improves) and early stopping (patience=2) to `train.py`, then retrained; the second run
correctly stopped at epoch 9 and kept the epoch-7 checkpoint (`val_mae=0.9794`,
`val_corr=0.7470`).

**Verification, including catching my own test-construction bugs**: `training/eval/
tactical_sanity.py` is a new permanent regression suite (material-imbalance sanity checks via
`eval_cli` + tactical search checks via `bestmove_cli`, formalizing what the original project
only ever did as one-off manual spot checks). Two of the five material test FENs were
initially hand-typed incorrectly (one was actually perfectly balanced material despite being
labeled "white up a queen"; another had an accidental bishop-for-rook imbalance masking the
intended rook deficit) — caught by cross-checking with `python-chess`'s piece counts rather
than trusting the hand-typed FENs, then fixed to construct test positions programmatically
(remove a piece from a known-good base position, assert the removed piece's type, let
`python-chess` validate the result) instead of typing FEN strings by eye. After the fix: 6/7
pass; the one soft-fail (a rook deficit scored at -252cp instead of the expected < -300)
correctly gets the *sign* right and is in a reasonable ballpark, a real but modest
calibration gap rather than a bug — while the queen-deficit case now nails the magnitude
almost exactly (+902cp vs. ~900cp classical value).

**Correlation improved measurably**: independently re-measured via `correlation_eval.py`
against held-out labels (not just trusting the training log), `pearson_r_clamped` went from
0.62 (M1, 400k positions, H=128) to 0.76 (M2, 6.1M positions, H=384+buckets), MAE from 235cp
to 194cp. Notably, this time the independent C++ measurement closely matches the Python
training log's own `val_corr` (0.75 vs 0.75), unlike M1 where there was a suspicious,
unexplained gap between the two — a sign the pipeline is more trustworthy this generation.

**Game-strength: measured honestly, including a false-positive result caught and corrected.**
An initial `quick_match.py` run at 300ms/move showed human_limit beating `old_engine` 75%
(7.5/10) — but a repeat run under identical settings minutes later showed 45%, and a third,
larger, more carefully-controlled 20-game run at 500ms/move (chosen specifically because
300ms is short enough that Windows process-launch jitter measurably changes which depth the
iterative-deepening loop completes, given how flat node-count growth is at these depths) came
back at **17.5% (3.5/20)** — a clear, real loss to `old_engine`. The 75% figure was noise from
a too-small sample at a too-short, jitter-sensitive time control; 17.5% on 20 games is the
number to trust. Recorded honestly rather than reporting the first (best-looking) number.

**Root-caused via direct measurement, not guessing**: queried the actual downloaded Stockfish
binary (not just the abstract "Stockfish project") on the same hardware/position/time budget.
At 1 second, Stockfish reaches depth 18-20 (effective branching factor ~1.94); `old_engine`
reaches depth 7 (branching factor ~5.8); `human_limit` reaches depth 5 (branching factor
~8.7) — despite Stockfish's raw nodes/sec being only 2.5x `old_engine`'s and ~10x
`human_limit`'s. The dominant gap is search-tree efficiency (pruning/move-ordering quality),
not eval speed: `old_engine` with Stockfish's branching factor and its *own current* node
budget would already reach depth ~18. This directly explains why the much-better-trained
`human_limit` eval still loses in real games — a smarter but more expensive per-node eval,
searched 2+ plies shallower with a worse branching factor, doesn't win. Dispatched a research
agent (results in `training/research_notes.md`, section 2, with live-Stockfish-source-verified
formulas for reverse futility pruning, forward futility pruning, late move pruning, and Static
Exchange Evaluation) and moved directly into implementing these in `engine/human_limit/
search.cpp` — see the next section of this log for what was actually built and verified.

## 13. Bitboard rewrite for human_limit, and the real bottleneck turned out to be elsewhere

Before implementing the pruning techniques from section 12's research, direct measurement was
requested: is the search-depth gap actually explained by the array-vs-bitboard board
representation, or something else? Isolated the question cleanly at depth 1 (zero possible
pruning for either engine, so depth-1 throughput is a pure measure of move-gen + eval cost).
Result: `human_limit` was only ~1.6-2.2x slower than the real downloaded Stockfish binary at
depth 1 — not the order-of-magnitude gap a purely architectural deficiency would suggest. The
node-count *growth* from depth 1 to depth 2 was the real tell: Stockfish's node count barely
moved (+13%), ours exploded (+441%), because our own null-move/LMR pruning has an explicit
`depth >= 3` gate — meaning depth 2 was never actually an apples-to-apples "no pruning"
comparison for Stockfish (its pruning is always active), while for us it genuinely was.

**Bitboard core, built additively, nothing removed.** Per explicit instruction, the existing
`chess/board.h/.cpp` and `chess/pieces.h/.cpp` (the array-based engine) were not touched at
all — they keep serving `ancient_engine` and `old_engine` exactly as before, and remain a
correctness oracle. A new `chess/bitboard/` module was built from scratch: `bitboard.h/.cpp`
(knight/king/pawn attack tables, a `squaresBetween` table for O(1) pin/check-block detection),
`magic.h/.cpp` (bishop/rook sliding attacks via magic bitboards — deliberately *generated* at
startup via random-trial-and-collision-check rather than copied from memorized published
constants, since a self-verifying generator can't produce an incorrect table by construction,
removing an entire class of "did I misremember the hex value" risk), and `position.h/.cpp` (a
new `Position` class mirroring `chess::Game`'s public surface, with pin/check bitboard
detection replacing the array engine's O(ray-length) loops with O(1) attack-table lookups).

**Staged, hard-gated verification before trusting any of it**, mirroring this project's
established discipline:
- Attack tables checked against independently-written brute-force ray-casting (not reusing
  the magic-search code's own ground truth) — 10,128 checks, 0 failures.
- Perft depth 1-5 from the standard start position plus three more standard reference
  positions (Kiwipete, and the classic "position 3"/"position 4" en-passant- and
  castling-heavy positions) — exact match on every single count, including 4,865,609 nodes at
  startpos depth 5 in under 2 seconds.
- Cross-validated move sets against `chess::Game::getValidMovesUciSlow()` (the existing
  oracle) across 2894 positions (30 random-walk seeds plus hand-crafted pin/double-check/
  en-passant-discovery positions) — 0 mismatches.
- A new bitboard-native accumulator path was added *alongside* (not replacing) the existing
  array-based one in `engine/human_limit/accumulator.h/.cpp` — `tests/test_accumulator.cpp`
  (array path) is untouched and still passes; `tests/test_accumulator_bb.cpp` (new) verifies
  the bitboard path matches an independent from-scratch recompute bit-exact across the same
  castling/en-passant/promotion battery used for the array version.

Only after all of that passed was `engine/human_limit/search.h/.cpp` retargeted from
`chess::Game`/`chess::Move`/`chess::UndoMove` to the new `Position`/`BBMove`/`BBUndo` types
(a mechanical swap given the matching interface), plus the CLI tools that construct positions
for the "human" engine branch. `ancient_engine`/`old_engine` source was not touched.

**Net result of the bitboard swap alone**: depth-1 throughput improved from 531 to 623 pos/sec
(+17%) — real, but far short of closing the gap to Stockfish. This was itself an important,
honestly-reported data point: it meant the board representation was only ever a modest
fraction of the real cost.

**The actual answer, found via a dedicated breakdown profiler
(`tools/profile_breakdown.cpp`) rather than sprinkling timers through production hot-path
code** (which would have added its own overhead and skewed the measurement): isolating move
generation, `makeMove`/`unmakeMove`, incremental accumulator updates, accumulator full
recompute, and the NNUE head forward pass as separate timed components showed the head
forward pass alone cost **26.07 microseconds per call — roughly 11x more than move generation,
make/unmake, and incremental accumulator updates combined (2.3us)**. All of the bitboard work
had been optimizing a piece of the pipeline that was only ever ~7% of the real cost.

**Two fixes, in order of what they actually bought:**
1. Flattened the NNUE head's per-bucket weights from `std::vector<std::vector<float>>` (8
   separately-heap-allocated, non-contiguous buffers) to a single flat contiguous
   `std::vector<float>` with computed per-bucket offsets, and switched `Accumulator` and the
   head forward pass from `double` to `float` throughout (matching the embedding table, which
   was already `float` — this also removes a float-to-double upconversion on every access).
   Measured effect alone: modest (head forward pass 26.07us -> 25.29us, ~3%) — informative in
   itself, since it showed memory layout wasn't the dominant cost.
2. **The real fix**: GCC will not auto-vectorize a serial floating-point reduction loop
   (`for (j) s += row[j] * x[j];`, the exact shape of every layer in the head) without
   `-ffast-math` or equivalent, because reordering floating-point additions is technically an
   IEEE-754 semantic change the compiler won't make unless explicitly permitted — regardless
   of `-O3 -march=native`. Confirmed directly by recompiling the same profiler with
   `-ffast-math` added: head forward pass dropped from 25.29us to **2.87us (8.8x)**, full eval
   from 30.6us to **7.0us (4.4x)**. Verified `-ffast-math` is safe project-wide first —
   `ancient_engine`/`old_engine` contain zero floating-point code (grepped to confirm), so the
   flag is a pure no-op for them; only `engine/human_limit/`'s NNUE code is affected.

**Combined, measured effect (not asserted) — the full turnaround**:
- Depth-1 throughput: `human_limit` 531 -> 623 (bitboards) -> **3268 pos/sec** (float+fastmath)
  — now **~3.75x faster than the real Stockfish binary's own depth-1 rate** (870 pos/sec) on
  this machine, a complete reversal from the ~1.6x slower starting point this session began
  investigating.
- At a fixed 1-second budget: `human_limit` now reaches **depth 7 at 335-368K nodes/sec**,
  matching `old_engine`'s depth exactly while exceeding its nodes/sec (250-256K) — `old_engine`
  numbers are unchanged throughout this section since it was never touched, staying on the
  array representation as the frozen baseline throughout.
- Full test suite (8 suites: array perft/legality, bitboard attack tables, bitboard perft,
  bitboard cross-validation, both accumulator paths, NNUE feature golden fixtures) all still
  pass after every change, including after switching `Accumulator` to `float` (tolerance
  loosened from 1e-6 to 1e-3 to accommodate legitimate float-vs-float reordering noise between
  incremental and from-scratch recompute — not a correctness weakening, since 1e-3 in the
  network's raw output scale is still ~0.4cp, far below any meaningful eval difference).
- Tactical sanity results (material recognition, back-rank mate-in-1, hanging-queen capture)
  are numerically identical before and after this entire section's changes — confirms the
  optimization changed speed only, not search/eval behavior.

Real game-strength confirmation (`quick_match.py`/`rating_ladder.py` against `old_engine`)
was in progress at the time of writing this section — see the next entry in this log for the
result once that run with the fully-optimized build completes.

That match finished after this section was written: **11.5/20 (57.5%)** for `human_limit`
vs `old_engine` at 500ms/move, up from 17.5% before the bitboard+float+fastmath work. This is
the confirmed result for section 13's changes, the last stable checkpoint before the pruning
work below.

## 14. Full pruning suite: SEE, RFP, LMP, futility, razoring, IIR, correction history,
singular extensions, ProbCut — deeper search, real-strength regression not yet resolved

Goal for this phase, stated explicitly by the user: no training, pure search/pruning work,
until `human_limit`'s search depth at a fixed 2-second budget is comparable to Stockfish's
depth at 2 seconds. Depth-1 throughput (raw per-node cost, no pruning involved) was never the
bottleneck — the gap is in how aggressively Stockfish prunes the tree per ply. Researched
current Stockfish internals directly from source (`training/research_notes.md` sections 2-4)
and implemented, in order:

- **SEE (Static Exchange Evaluation)** — `chess::bitboard::Position::see()`, recursive
  swap-list algorithm (`seeSwing`), evaluates a capture's net material result assuming best
  play by both sides on the target square, using `attackersTo()`/x-ray removal via the
  `squaresBetween` snipers technique. Used for capture move-ordering (losing captures sorted
  below quiets) and for pruning clearly-losing captures in quiescence search.
- **Reverse futility pruning (RFP)** — depth ≤ 8, prune when static eval already exceeds beta
  by a depth-scaled margin (150cp/ply, tightened when `improving`).
- **Forward futility pruning** — depth ≤ 3, skip quiet moves when static eval + margin can't
  reach alpha.
- **Late move pruning (LMP)** — depth ≤ 6, skip late quiet moves once move count exceeds a
  depth/improving-scaled threshold. Both futility and LMP are gated so the first move at a
  node is never pruned.
- **Razoring** — depth ≤ 3, drop straight to quiescence when static eval is far below alpha,
  verified rather than trusted blindly.
- **Internal iterative reduction (IIR)** — depth ≥ 6 with no TT move: reduce depth by 1 before
  searching, since the absence of a TT move at that depth suggests the position wasn't
  well-explored previously.
- **The `improving` flag** — is the current position's static eval better than 2 plies ago
  (same side to move)? Sharpens RFP/LMP/futility margins when true.
- **Correction history** — one structural-hash table (`corrHist_[2][16384]`, scoped down from
  Stockfish's four-table pawn/minor/non-pawn/continuation split) that learns a per-position-type
  bias correction applied to static eval, updated from search results each node.
- **Singular extensions + multi-cut** — at depth ≥ 6 with a trustworthy TT entry, re-search
  excluding the TT move at a reduced window; if no other move can beat that window, the TT
  move is "singular" and gets extended; if multiple moves already beat beta, cut early
  (multi-cut).
- **ProbCut** — depth ≥ 5, a shallow reduced-depth search with a margin above beta to detect
  positions where a null-window search will almost certainly fail high, cutting without the
  full-depth search.

Per explicit user instruction ("do not overwhelm files with incredible amounts of comments"),
`search.cpp`/`search.h` carry brief comments only where the constant or gate isn't
self-explanatory (e.g. why IIR triggers, what `improving` means) — not per-line narration.

**Verification.** All 9 existing test suites (array perft/legality, bitboard attack tables,
bitboard perft, bitboard cross-validation, both accumulator paths, NNUE feature golden
fixtures, SEE hand-traced cases) still pass. `tactical_sanity.py` is numerically unchanged at
6/7 (the pre-existing rook-deficit soft-fail is untouched — confirms this phase didn't change
eval, only search shape).

**Depth/EBF measurement (2-second budget, fixed positions):**
- Before this phase: `human_limit` depth 8, effective branching factor (EBF = nodes^(1/depth))
  ≈ 4.6.
- After this phase: `human_limit` depth 9, EBF ≈ 4.06-4.08 — real improvement in tree
  efficiency, but still well short of the phase's stated goal.
- Stockfish at the same 2-second budget: depth 21-23, EBF ≈ 1.82-1.93. The gap did not close
  to "comparable" — Stockfish is still pruning roughly twice as efficiently per node, on top of
  its own raw per-node speed advantage. This goal is **not yet reached**; more pruning work is
  warranted if the user wants to keep pursuing it.

**Open, unresolved concern — pruning quality vs. depth.** The user explicitly raised the risk
that these margins (RFP 150cp/ply, LMP thresholds, ProbCut's 200cp margin, singular-extension
verification windows, etc.) are hand-picked from research rather than SPRT-tuned like real
Stockfish's, and that chasing nominal depth this way could cost genuinely good moves that get
pruned away. This is a live, real risk, not yet dismissed: a `quick_match.py` 20-game run
against `old_engine` with the full pruning suite in production, at the same 500ms/move setting
used for the confirmed 57.5% baseline above, opened at a concerning 2.0/8 (25%) after 8 games —
well below baseline — before trending back up to roughly 4.0/11 (36%) by game 11, still below
the 57.5% baseline and still in progress at the time of writing. Whether the final number lands
near baseline (small-sample noise, consistent with this project's established pattern of
high-variance short-time-control results) or confirms a real regression from the untuned
pruning constants is not yet known. **This entry will be updated with the final score once the
match completes; if it confirms a real regression, the most likely first culprits are forward
futility and LMP, since both trust static eval directly to skip moves rather than verifying via
a reduced search the way razoring and RFP do.**

The `old_engine` win-rate match started above was interrupted (stopped by the user) before
finishing. The user then clarified the evaluation approach going forward: `human_limit`'s NNUE
is not yet trained against Stockfish and will obviously play weakly against `old_engine` until
that training happens, so win-rate against `old_engine` is not a meaningful signal for judging
pruning work right now — it can't distinguish "search got shallower" from "eval is still
untrained." **Pruning quality from here on is judged by search depth reached at a fixed time
budget (0.5s/1s/2s/3s) compared to Stockfish, not by win rate.** Win-rate testing against
`old_engine` returns once training resumes after pruning work is done.

## 15. Pruning rewritten from scratch (v2), old pruning removed; zobrist hash bug found and
fixed; 5x-speedup retested and confirmed unaffected

Per the user's direction, the pruning implementation from section 14 (call it v1: SEE, RFP,
LMP, futility, razoring, IIR, correction history, singular extensions, ProbCut, with a flat
null-move reduction and a flat late-move-reduction) was first preserved untouched in a
separate file, then rewritten from scratch aiming at the highest-leverage techniques research
identifies for closing the depth gap to Stockfish, and finally the old file was deleted once
v2 was confirmed working — `human_limit` now relies solely on the v2 pruning implementation
in `engine/human_limit/search.cpp`/`search.h`.

**What v2 changes and why**, in order of expected impact (per the research: null-move pruning
and late-move reductions are by far the largest levers, at "roughly 100-150 Elo"/"50-80% node
savings" and "often halves branching" respectively — v1's implementations of both were the
most conservative, generic versions):

- **Null-move pruning**: v1 used a flat reduction of R=2 regardless of depth. v2 uses a
  depth- and eval-scaled reduction (`R = 3 + depth/4 + min((staticEval-beta)/200, 3)`, closer
  in shape to Stockfish's `R = 7 + depth/3`, scaled down for our shallower depths), gated on
  `staticEval >= beta` (only take the "pass and still winning" bet when actually ahead), plus
  a verification re-search (null-move disabled) at depth ≥ 12 to catch zugzwang positions that
  a depth-only guard can miss.
- **Late move reductions**: v1 used a flat `depth-2` reduction for any quiet move past index 3
  at depth ≥ 3. v2 uses a precomputed log-based table, `r = 1 + c*ln(depth)*ln(moveIndex)`,
  mirroring the shape of Stockfish's actual formula — later moves at higher depth get reduced
  much more aggressively than v1's flat cut allowed, while early late-moves are barely reduced.
- **Continuation history** (new): a `[prevPiece,prevTo][piece,to]`-indexed table tracking how
  well a quiet move has worked as a reply to the specific piece that just moved — layered onto
  the existing butterfly history for move ordering, sharper than butterfly history alone since
  it's context-sensitive rather than one global score per (from,to).
- **History/continuation-history gravity** (new): quiet moves that were tried at a node but
  didn't end up causing the cutoff now get a symmetric malus, not just the cutoff move a bonus
  — keeps the tables from drifting upward without bound and improves ordering precision.
- **Mate distance pruning** (new): tightens alpha/beta at each node to the range no mate score
  through that node could exceed, letting an already-found short mate cut off remaining
  branches for free.
- SEE, RFP, forward futility, LMP, razoring, IIR, correction history, singular extensions, and
  ProbCut are carried over from v1 unchanged — they tested clean and target different parts of
  the tree than NMP/LMR, so there was no reason to touch them this pass.

**Bug audit.** Per the user's request, a separate read-only agent reviewed the bitboard chess
core (`chess/bitboard/*`, `engine/human_limit/accumulator.*`, `engine/human_limit/network.*`)
that produced the earlier 5x eval speedup, purely as a sanity check — not part of the pruning
work. It found one real, confirmed bug:

- **Zobrist hash omitted castling rights and en-passant state** (`chess/bitboard/position.h`/
  `.cpp`). Two positions with identical piece placement and side to move, but different
  castling rights or en-passant availability, hashed identically — this could cause false
  repetition-draw claims and transposition-table collisions returning a cached score/move
  computed under different castling/en-passant conditions. Concrete failure case: from the
  start position, `1. Rb1 Rb8 2. Ra1 Ra8` returns to the starting piece placement with White
  to move, but with queenside castling rights permanently lost on both sides — the old hash
  couldn't tell this apart from the true start position.
  - **Fix**: added 4 zobrist keys for castling rights and 8 for the en-passant file, toggled
    in `makeMove`/`makeNullMove` whenever castling rights or en-passant availability change
    (unmake already restores the pre-move hash verbatim from `BBUndo`/`NullUndo`, so no
    unmake-side change was needed).
  - The audit's other flag (SEE not crediting promotion value mid-exchange) was assessed as a
    standard, accepted simplification shared by most engine SEE implementations, not a bug —
    left as-is.
  - Magic bitboard generation, move generation (castling-through-check, en-passant discovered
    check, pins, promotions), and the float/`-ffast-math` conversion were all reviewed and
    found correct — no changes made there.

**Verification after both changes** (pruning rewrite + zobrist fix): `test_bitboard_attacks`
(10,128 checks), `test_bitboard_perft` (start/Kiwipete/position3/position5, all exact),
`test_bitboard_cross_validate` (2,894 positions, 0 mismatches), and `test_see` (5 hand-traced
cases) all still pass. The full `bestmove_cli` binary builds clean and produces legal moves on
spot-check positions (one apparent crash during testing, "king not found", was traced to an
invalid test FEN with two kings adjacent — an illegal chess position, not an engine bug;
confirmed by reproducing the same crash against the pre-rewrite v1 pruning code with the same
FEN).

**5x-speedup retest.** Since the zobrist fix touched `position.cpp` — part of the code that
produced the earlier eval/throughput speedup — it was retested to confirm no regression.
`throughput_depth2` (500 random positions, depth 1, batched in-process — no IPC overhead):
`human_limit` unchanged at **2890 pos/sec** (vs `old_engine`'s 403 pos/sec, 7.2x — consistent
with pre-fix numbers). A fresh apples-to-apples comparison against Stockfish was also run this
time (persistent UCI process, 500 identical positions, `go depth 1`, to remove per-call
subprocess-launch overhead that likely inflated earlier informal comparisons): Stockfish
reached **2729 pos/sec** under the same harness — i.e. **roughly at parity with `human_limit`
here (~1.06x), not the ~5x figure quoted earlier in this log.** The zobrist fix did not cause
this — `human_limit`'s own number is unchanged before/after the fix. The discrepancy is most
likely that earlier "5x" figures came from a narrower per-eval-call microbenchmark rather than
this full position-throughput harness, which also carries Stockfish's own UCI-layer overhead.
Treat this section's number as the current best apples-to-apples baseline going forward.

**Depth-at-fixed-time vs Stockfish** (single middlegame position, single-threaded Stockfish,
persistent process for both engines — this is now the primary pruning-quality metric per the
user's direction, not win rate):

| time | `human_limit` depth | Stockfish depth |
|---|---|---|
| 0.5s | 9 | 17 |
| 1s | 10 | 18 |
| 2s | 11 | 22 |
| 3s | 12 | 24 |

Improved from the pre-v2 8-9 depth at 2s, but still roughly half Stockfish's depth at every
time budget — the "get close to Stockfish's depth at these times" goal is **not yet reached**.
Next candidates, per the research notes, are more aggressive move-count-based LMR (Stockfish
reduces much harder at high move index than v2's table currently does) and capture history for
sharper capture ordering beyond plain SEE.

**Reference: Stockfish's own time-to-depth curve, on both an easy and a hard position** — kept
here so future `human_limit` depth numbers can be compared against a consistent baseline
without re-running Stockfish each time. Both runs are single-threaded, run to depth 25, timed
from the first `go` command (persistent UCI process, no per-call subprocess overhead).

*Easy/early-middlegame position* (`r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w
KQkq - 4 4` — the same position used for the `human_limit` vs. Stockfish depth-at-time table
above):

| depth | time | nodes |
|---|---|---|
| 1-6 | ≤12ms | 43 → 1,779 |
| 7-12 | 23-60ms | 5,954 → 22,494 |
| 13 | 197ms | 66,902 |
| 14 | 280ms | 114,185 |
| 15 | 318ms | 138,156 |
| 16 | 446ms | 220,632 |
| 17 | 513ms | 255,723 |
| 18 | 915ms | 449,192 |
| 19 | 1.26s | 640,517 |
| 20 | 1.90s | 997,857 |
| 21 | 2.67s | 1,393,802 |
| 22 | 2.82s | 1,439,756 |
| 23 | 4.22s | 2,001,543 |
| 24 | 4.84s | 2,345,175 |
| 25 | 5.19s | 2,516,853 |

*Complex, open, tactically live middlegame* (`r1b2rk1/1pq1bppp/p1n1pn2/3p4/2PP4/1PN1PN2/
PB1QBPPP/R4RK1 w - - 0 1` — both sides fully developed and castled, queens and both rook pairs
still on the board, central tension, high piece mobility — chosen specifically to be harder
than an opening/early-middlegame position):

| depth | time | nodes |
|---|---|---|
| 1-9 | ≤5ms | 55 → 2,133 |
| 10 | 8ms | 4,186 |
| 11 | 10ms | 5,125 |
| 12 | 18ms | 8,790 |
| 13 | 26ms | 12,758 |
| 14 | 49ms | 22,384 |
| 15 | 196ms | 66,546 |
| 16 | 436ms | 165,721 |
| 17 | 651ms | 260,393 |
| 18 | 1.13s | 499,821 |
| 19 | 1.34s | 590,852 |
| 20 | 1.76s | 769,129 |
| 21 | 2.41s | 1,017,468 |
| 22 | 4.61s | 1,962,961 |
| 23 | 6.30s | 2,638,302 |
| 24 | 8.89s | 3,617,582 |
| 25 | 16.77s | 5,855,173 |

Takeaways: even Stockfish's node counts scale up meaningfully on the harder position from
depth ~20 onward (e.g. depth 22 nearly doubles depth 21's node count here, vs. a much smoother
climb on the easier position) — complexity shows up as a steeper *late*-depth cost, not a
different shape at shallow depths (both positions are still "free," under 20ms, through depth
~12-13). This matters for `human_limit` comparisons: our depth-at-time numbers so far (section
14/15 above) were measured on the easy position; once pruning work continues, depth-at-time
should also be checked against this harder position, since a technique that looks fine on an
open, low-tension position can behave differently once real tactical density shows up.

## 16. `human_limit` v2 baseline time-to-depth vs Stockfish, and the three-milestone plan

Established `human_limit`'s own time-to-depth curve (per-ply timing added to `findBestMove`'s
iterative-deepening loop, gated behind `HL_VERBOSE=1` so it prints to stderr only when set —
no production cost). Measured on the exact two positions used for the Stockfish curves in
section 15 above, single-threaded, TT carried across depths within one ID run (fair comparison
to Stockfish's ID). Head-to-head:

**Easy position** (`r1bqkb1r/...RNBQK2R w KQkq - 4 4`):

| depth | Stockfish time / nodes | human_limit time / nodes | node ratio |
|---|---|---|---|
| 10 | 37ms / 12,507 | 614ms / 90,991 | 7.3x |
| 12 | 60ms / 22,494 | 2,864ms / 447,715 | 19.9x |
| 14 | 280ms / 114,185 | 7,995ms / 1,271,558 | 11.1x |
| 16 | 446ms / 220,632 | 18,199ms / 2,855,320 | 12.9x |

**Hard position** (`r1b2rk1/1pq1bppp/p1n1pn2/3p4/2PP4/1PN1PN2/PB1QBPPP/R4RK1 w - - 0 1`):

| depth | Stockfish time / nodes | human_limit time / nodes | node ratio |
|---|---|---|---|
| 10 | 8ms / 4,186 | 1,347ms / 237,524 | 56.7x |
| 12 | 18ms / 8,790 | 7,071ms / 1,156,326 | 131.5x |
| 14 | 49ms / 22,384 | 17,296ms / 2,864,521 | 128x |

**The decisive finding — the gap is almost entirely node count (EBF), not per-node speed, and
it is dramatically worse on complex positions.** Our nps is a stable ~160K regardless of
position (easy depth 14: 159K nps; hard depth 14: 165K nps), while Stockfish runs ~460-490K
nps — a ~3x per-node speed gap (eval cost, a separate lever, not pruning). But that ~3x is
dwarfed by the node-count gap: on the hard position at depth 12 we search **131x** more nodes
than Stockfish. Decomposing the 393x wall-time gap at hard-position depth 12: ~3x is nps, the
remaining ~130x is purely searching too many nodes. **That ~130x is entirely a
pruning/move-ordering problem — exactly the lever we're working on.**

Per-ply EBF (node multiplier per depth): Stockfish ~1.3-1.5 on both positions. `human_limit`
~1.7-1.8 on the easy position (not far off), but **~2.0-2.3 with a spike to 3.6x (depth 10→11)
on the hard position.** Halving that EBF (1.8→1.4) compounds enormously over 12+ plies. Our
ordering/reductions hold up on quiet positions but break down badly under tactical density —
the classic signature of move ordering failing (best move not tried first, so alpha-beta stops
cutting) and reductions being too timid where there are many candidate moves.

**Three-milestone plan (user-directed), targets set against the Stockfish curves above:**
1. **"Free computation"** (top priority): reach ~depth 12 in tens of ms, matching Stockfish's
   under-60ms-to-depth-12. Dominated by the node-count gap; almost pure pruning/ordering work.
   The complex-position EBF explosion is the specific thing to kill first.
2. **"First 2 seconds"**: depth 20 in ~2s (Stockfish reaches depth 20 in ~1.8s easy / ~1.76s
   hard). Second priority.
3. **"Vast expanse"**: depth 25+ in the 5s+ range (Stockfish: ~5s easy / ~17s hard to depth
   25). Third priority.

The residual ~3x nps gap (eval cost) is noted as a separate, later lever — it is not on the
pruning path and does not block these milestones, since node-count wins of 10-100x are
available first and are what the milestones actually hinge on.

## 17. Milestone-1 pruning improvements (running log, each measured against §16 baseline)

Each change below is applied one at a time, rebuilt, and measured on both §16 positions via
`HL_VERBOSE=1 diag_speed <fen> 30000 16` (time-to-depth to 16), kept only if node count drops
without tactical quality loss. Baseline (before any of these) = §16 numbers.

**(1) Aggressive LMR + LMR on losing captures.** v2's initial LMR reduced only quiet,
non-checking moves starting at move index 3, with table `r = 0.9 + 0.55·ln(d)·ln(mi)`. Changed
to: table `r = 1.0 + 0.75·ln(d)·ln(mi)` (about +1 ply reduction at mid/high depth-index), start
at move index 2, and extend LMR to SEE-losing captures (with an extra +1 reduction) — tactical
positions were previously searching every capture at full depth, which is most of the tree
there. Result (time / nodes to reach the depth):

| | baseline d12 | now d12 | baseline d14 | now d14 |
|---|---|---|---|---|
| easy | 2,864ms / 448K | **902ms / 156K** | 7,995ms / 1.27M | **2,577ms / 423K** |
| hard | 7,071ms / 1.16M | **4,257ms / 690K** | 17,296ms / 2.86M | **11,407ms / 1.87M** |

~2.9-3.0x fewer nodes on the easy position, ~1.5-1.7x on the hard position. Iteration scores
unchanged/sane. Tactical safety spot-checked: the queen-sacrifice combination `1k1r4/pp1b1R2/
3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - -` still finds `Qd1+` (d6d1) — a surface-losing-capture sac,
exactly the move type aggressive LMR + losing-capture reduction risks pruning; the PVS
full-depth re-search net catches it. Kept.

Note: the depth 10→11 node spike on the hard position (§16) persists and is proportionally
unchanged (~4x), so it is *not* an LMR issue — it correlates with an inter-iteration score
drop, i.e. search instability / a refutation appearing at that depth.

**(2) History-based pruning + history-adjusted LMR — REVERTED (net negative).** Tried:
raise LMP depth ceiling 6→8, add history leaf pruning (skip quiets with strongly negative
history+contHist), and scale LMR reduction by move history. Result: node count went *up* at
higher depths (easy depth 16: 1.22M → 2.09M) with volatile scores (easy depth 16 score 30→13)
— the history signal from the untrained/noisy eval is not reliable enough to prune/reduce on,
only to order with. Reverted in full.

**(3) Transposition table 1M → 4M entries — KEPT (marginal win).** Instrumented aspiration
re-searches first (`aspFails` in the verbose line): **0 at every depth on both positions**, so
the hard-position spike is *not* an aspiration re-search — it is genuine tree growth from PV
instability. But a bigger TT did help: the hard depth-11 spike dropped 527K → 361K nodes
(-31%), and easy high-depth improved (depth 16: 1.22M → 830K, -32%). Neutral-to-slightly-worse
on hard at depth 12-16. Kept for the spike/easy wins; costs ~128MB. (A depth-preferred +
aging replacement policy was also tried on top and was *worse* than plain always-replace —
always-replace keeps fresher move-ordering info, which matters more here than protecting deep
entries — so replacement policy was left as always-replace.)

**(4) Capture history for capture ordering — REVERTED (not robust).** Ordering-only (safe by
construction, can't miss moves), indexed by [movingPiece][to][capturedType]. Helped the hard
position (depth 12: 693K → 597K) but hurt the easy one (depth 12: 152K → 237K) and added score
volatility, at both a strong (ch/32) and light (ch/128) blend weight. Trades easy for hard
without a net win — same root cause as (2): the noisy untrained eval keeps shifting the PV, so
history-derived signals don't converge. Reverted.

**Net kept from this milestone-1 pass: (1) aggressive LMR + losing-capture reduction, and (3)
the 4M TT.** Cumulative vs the §16 baseline:

| | baseline d12 | now d12 | baseline d14 | now d14 | baseline d16 | now d16 |
|---|---|---|---|---|---|---|
| easy | 2,864ms / 448K | **1,150ms / 152K** | 7,995ms / 1.27M | **2,198ms / 311K** | 18,199ms / 2.86M | **5,575ms / 830K** |
| hard | 7,071ms / 1.16M | **4,136ms / 693K** | 17,296ms / 2.86M | **11,212ms / 1.90M** | — | 23,766ms / 4.01M |

~3-4x fewer nodes on the easy position, ~1.5x on the hard one. Correctness gates (perft, SEE,
cross-validate) all still pass; `Qd1+` queen-sac tactic still found.

**Key diagnosis for the road ahead.** The hard-position node count (693K at depth 12 vs
Stockfish's 8.8K = 79x) is the wall, and it is driven by *search instability*: the score
oscillates across iterations (hard depth 10→16: 89, 80, 83, 84, 86, 90, 94), the PV keeps
changing, and each change forces the tree to re-establish a new best line. Three separate
standard techniques that normally cut this (history pruning, depth-preferred TT, capture
history) each either hurt or washed out here — the common thread is that they all rely on
signals (history, eval-derived scores) that are unreliable because **the NNUE eval is still
untrained and noisy**, so it keeps changing its mind about the best line. This strongly
suggests the remaining large chunk of the hard-position gap is gated on the eval training the
user has deferred, not on more pruning heuristics — pruning has picked the clean, eval-noise-
independent wins (LMR, TT) available at this stage. Aspiration windows are confirmed *not* a
factor (aspFails=0 throughout).

## 18. Fresh NNUE training run: Stockfish depth-2 teacher (user's hypothesis), 6h budget

Per §17's diagnosis (pruning is now bottlenecked on the untrained/noisy eval, not on more
pruning heuristics), the user chose to pursue training next rather than continue the pruning
list. The user's specific hypothesis: at inference time the NNUE only ever produces a *static*
evaluation (no search of its own) at leaf nodes, so training it against a teacher that has
searched 14+ plies deep may be teaching it to approximate a signal it structurally cannot
reproduce from a single position snapshot. Training against a teacher limited to **depth 2**
(one that mostly "sees what's there" rather than "what's going to happen many moves out") may
be a better match for what a static evaluator can actually learn. This has not been tried
before in this project — worth a real, isolated test.

**Experimental design.** To isolate teacher-depth as the only variable (vs. the current
production model), the exact same FEN corpus that trained the current model was reused rather
than generating new positions: `training/data/raw/lichess_6m.jsonl` (6.0M real Lichess-game
positions) + `training/data/raw/curriculum_labeled.jsonl` (122,607 curriculum positions:
self-play, post-capture, material-imbalance, endgame) = 6,122,607 FENs, all relabeled fresh
with Stockfish at `depth=2` (new script `training/data/relabel_shallow.py`, output
`training/data/raw/shallow2_labeled.jsonl`). Same position distribution, only the teacher's
search depth changes — a clean single-variable test.

**Bug found and fixed in existing labeling infra.** `training/data/stockfish_teacher.py`'s
`_label_worker` called `engine.quit()` *before* `out_queue.put(results)`. Under worker-count
oversubscription (tested 14 workers on a 6-core machine to probe scaling) some Stockfish child
processes died before `quit()`, `quit()` raised `EngineTerminatedError`, and because `put()`
had not run yet, the parent's `queue.get()` blocked forever waiting for a result that would
never arrive — a silent hang, not a crash. Fixed by reordering (`put()` before `quit()`) and
wrapping `quit()` in a try/except so a dead engine can't prevent already-computed results from
being returned. Confirmed fixed and re-benchmarked at safe worker counts: 6 workers ≈
1700-1775 pos/sec sustained (8 workers gave no improvement — 6 cores is the real ceiling on
this machine), no hangs.

**Pipeline**, running now as a self-handoff background job (`relabel_shallow.py` →
`build_dataset.py` → `train.py`, chained via a small orchestrator script so no manual
babysitting is needed between stages):
1. Relabel all 6,122,607 FENs at Stockfish depth=2, 6 parallel workers (~55-65 min).
2. Build `training/data/processed/shallow2_dataset.npz` from the relabeled data (same
   `build_dataset.py`/`encode.py` pipeline as every prior dataset — no format changes).
3. Train a fresh network (`training/model/train.py`, same architecture: hidden=384, 8 output
   buckets, WDL-space loss) against a 6-hour total wall-clock budget. Added a `--max-minutes`
   wall-clock cap to `train.py` (previously only epoch-count/early-stopping governed length)
   so a long unattended run has a hard stop regardless of how epochs trend; set to 290 minutes
   here (~4.83h), leaving the relabeling time inside the user's 6h ask (~55-65min + 290min ≈
   5.7-5.9h). Early-stop patience raised from the usual 2 to 6 for this longer run, since a
   6-hour budget can afford to ride out a longer plateau before giving up. Best-checkpoint
   saving (on `val_mae` improvement) is unchanged from prior runs, so the final checkpoint is
   never worse than what the wall-clock or early-stop cutoff caught mid-plateau.

**Revised, larger plan.** The initial framing above (train one 384-hidden network for up to
4.83h) was reconsidered before training started: this NNUE is small and converges fast on CPU
(the prior production run hit its best checkpoint at epoch 7), so a single run would likely
early-stop within well under an hour, leaving most of the 6h budget idle. Two changes were
made to actually use the time:

1. **Bigger network.** Hidden width raised 384 → **1024** ("Stockfish-sized" per the user's
   request, i.e. meaningfully closer to real NNUE scale rather than a token bump) for the
   phase-1 model trained on the full base corpus.
2. **A continuous Stockfish-teacher self-play phase after the base training**, per the user's
   own design: rather than stopping after the 6.1M-position base corpus, keep generating fresh
   positions forever using the pattern **"play 2 depth-2 best moves, then 1 random legal
   move," repeating** — the best-move steps keep games on sound, realistic lines (matching
   what the depth-2 teacher itself would choose), while the random move cheaply injects
   positional variance/off-book positions, avoiding a pure random-walk's unrealistic games.
   Each visited position (both post-bestmove and post-random) is labeled with the exact
   depth-2 evaluation already computed to select that step's move — free, no extra Stockfish
   calls (`training/data/selfplay_stream.py`, new). Smoke-tested at 414 pos/sec per worker.

**New orchestrator** (`training/orchestrate_full.py`, replaces the earlier shell-only chain):
1. Wait for the depth-2 relabeling of the base 6.1M corpus to finish (unchanged from above).
2. Build the base dataset, train the hidden=1024 network on it (capped at 60 min — this is
   the "does the depth-2-teacher hypothesis work at all" checkpoint).
3. Launch 3 parallel self-play workers (leaving headroom on the 6-core machine for the
   training process itself) for the remaining budget, each running the 2-best-then-random
   pattern continuously and streaming labeled positions to its own file.
4. Every 25 minutes, take only the *newly generated* positions since the last cycle (not the
   whole accumulated pool — keeps each cycle's dataset-build cost bounded rather than growing
   without limit), build a small dataset from just that delta, and warm-start fine-tune the
   current checkpoint on it (`--init-from`, new `train.py` flag added this session) at a lower
   learning rate (3e-4 vs the base run's 2e-3) for a few epochs — standard continual
   fine-tuning shape, low LR + few epochs per increment to avoid catastrophic forgetting of
   the base corpus's broad distribution while still absorbing fresh Stockfish-quality data.
5. Stops self-play workers near the budget's end, does one final consolidation fine-tune over
   everything self-play generated, and writes the winning checkpoint path to
   `training/checkpoints/LATEST_SHALLOW2.txt`.

Total budget ~330 min for this orchestrator (accounts for time already spent getting the
relabeling/setup going), with a 20-min final buffer reserved for the consolidation pass.

**Bug fixed in passing**: while testing worker-count scaling on the labeling infra, found and
fixed a real hang bug in `stockfish_teacher.py`'s `_label_worker` — it called `engine.quit()`
*before* `out_queue.put(results)`, so if the engine process died before `quit()` (observed
under 14-way oversubscription on this 6-core machine), the exception meant `put()` never ran
and the parent's `queue.get()` blocked forever. Fixed by reordering and wrapping `quit()` in a
try/except. Re-verified safe worker counts (6 ≈ 1700-1775 pos/sec sustained at depth=2; 8
workers showed no further gain — 6 cores is the real ceiling here).

**Order flipped per user direction, before either phase had actually started training.** The
user asked to run phase 2 (continuous self-play generation + incremental fine-tuning) *first*,
for about 4 hours, establishing the base model from scratch — then run phase 1 (the deep
6.1M-position corpus) *second*, afterward, as a high-quality fine-tuning pass on top of the
self-play-trained model, rather than the other way around. `training/orchestrate_full.py` was
rewritten accordingly: `phase2_selfplay_and_finetune()` now runs first (240 min budget),
initializing the hidden=1024 network from scratch on its first 25-minute cycle and warm-start
fine-tuning on each subsequent cycle's fresh self-play data; only after that completes does it
wait for the (already-in-progress, unaffected by any of this reordering) depth-2 base-corpus
relabeling, build that dataset, and run one final warm-started fine-tune pass over the full
6.1M real-game corpus at a lower learning rate (5e-4) as the polishing step, writing
`training/checkpoints/shallow2_final.pt`.

**A second real bug found and fixed before launch**: native Windows Python resolves a path
like `/tmp/relabel.log` as drive-root-relative (`C:\tmp\relabel.log`), *not* the same location
as bash's `/tmp` (which maps to `C:\Users\notu7\AppData\Local\Temp` on this machine, confirmed
via `cygpath -w`). The orchestrator's `wait_for_relabel()` was checking the wrong path
entirely and would have polled forever without ever finding the relabeling log the bash-side
job was actually writing to — a silent infinite wait, never triggering phase 1. Fixed by
adding an explicit `TMP_DIR`/`tmp()` helper in `orchestrate_full.py` pointing at the real
mapped path, used consistently for every cross-language log/status file reference. Caught by
checking actual file existence after the first launch attempt rather than assuming a clean
process start meant a working pipeline — worth remembering as a recurring class of bug in this
environment (this project has hit Windows/bash path-mapping mismatches at least twice before,
per earlier session notes on `subprocess`/relative-path issues).

**Superseded.** The flipped-order 2-phase plan above was stopped entirely (`kill all training
that is happening right now`) partway through phase 2's self-play generation, and the design
was iterated on further via a series of sample-game inspections before settling on a final
4-phase plan (below). The base relabeling job was also killed mid-run at that point — it had
reached 4,042,063/6,122,607 positions (depth=2) before being stopped; that partial file
(`training/data/raw/shallow2_labeled.jsonl`) was kept and reused as phase 1's training data
below rather than discarded.

## 19. Self-play schedule design, iterated via direct game inspection

Before committing compute to a long run, the self-play generation schedule
(`training/data/selfplay_stream.py`) was refined through several rounds of "generate a sample
game, read it, adjust" — cheaper than discovering a bad design 3 hours into an unattended run:

- **Games previously stopped at a 120-ply cap without reaching a real conclusion.** Changed to
  play to actual game-over (`board.is_game_over(claim_draw=True)`), with a 300-ply safety net
  only for pathological non-terminating cases.
- **Random-move cadence tuned down twice**: every-3rd-ply → every-4th-ply → **every 9th ply**
  (8 "best" moves then 1 random, repeating). The period was chosen to be **odd on purpose**:
  since ply parity determines side to move, an odd period means each successive random move
  lands on the opposite color from the last, so both sides draw random moves roughly equally
  over a game — confirmed directly in generated samples (random moves alternating White/Black:
  ply 8, 17, 26, 35, 44 in one sample).
- **Random injection now stops after ply 50** — past that point the game plays pure best-teacher
  moves so it actually reaches a decisive or drawn conclusion instead of being perturbed forever.
- **Endgame search depth** (past ply 50) set to depth 7 (tried depth 10 first, settled on 7 for
  cost reasons — see phase 3/4 below for why depth 7 specifically was later revisited).
- Added an explicit **opening schedule**: plies 0-3 forced random (pure opening variance), plies
  4-7 forced best-move at depth 7 (settles the position after the random opening) before the
  depth-2/random-every-9th middlegame phase takes over at ply 8.
- **Generation vs. grading depth were explicitly decoupled.** Initially the same `analyse()`
  call was reused both to pick the move and to produce the training label. Per the user's
  clarification ("game generation stays same... what do we do with the games we generate? We
  GRADE THEM WITH STOCKFISH"), `selfplay_stream.py` now does two separate engine calls per
  visited position: one at the schedule's move-selection depth (2/7/2/7 by phase, unchanged),
  and a second, independent `GRADE_DEPTH=4` call whose score is what actually gets written out
  as the training label. This roughly doubles per-position engine cost but means the label
  quality is a free variable, decoupled from whatever depth was cheapest for move selection.

**Throughput measurements** (single worker unless noted, this 6-core machine):
- Isolated depth-2 eval only (no game logic): 880/sec.
- Isolated depth-4 eval only: 611/sec.
- Full self-play generation (schedule above, no separate grading, i.e. reusing the
  move-selection eval as the label): ~520/sec single-worker.
- Full self-play generation **+ separate depth-4 grading pass** (the design actually used):
  **275.5/sec single-worker → ~826.5/sec combined at 3 workers → ~1.49M graded positions per
  30 minutes at 3 workers.**

## 20. Final 4-phase training plan (current, running)

Per explicit user instruction, replacing all prior plans:

1. **Phase 1**: train the initial hidden=1024 network on the ~4.04M positions already
   depth-2-graded from the killed relabeling run (`shallow2_labeled.jsonl`) — reuses that
   partial work rather than discarding it.
2. **Phase 2** (3 hours): continuous self-play generation using the schedule from §19 (3
   parallel workers, each visited position graded at depth=4 via the decoupled grading pass),
   with incremental warm-started fine-tune cycles every 25 minutes folding in that cycle's
   fresh data (low LR 3e-4, few epochs, matches the continual-fine-tuning shape used in the
   earlier 2-phase design).
3. **Phase 3**: regrade the *entire* 6,122,607-position base corpus (`lichess_6m.jsonl` +
   `curriculum_labeled.jsonl`) at depth=4 from scratch (`shallow4_labeled.jsonl`) — supersedes
   the partial depth=2 file used for phase 1, now used only as phase 1's seed data.
4. **Phase 4**: final fine-tune of the phase-2 checkpoint on the depth-4-regraded 6.12M corpus
   (real-game positions, deeper/consistent grading), producing
   `training/checkpoints/final_model.pt`.

New orchestrator: `training/orchestrate_4phase.py` (replaces `orchestrate_full.py`).

**A real timing risk was checked and resolved before trusting the pipeline.** hidden=1024 is
~2.7x the compute of the prior 384-hidden network; phase 1's first epoch on the full 3.84M-row
training split took **1,312s (~21.9min)**, raising a genuine concern that phase 2's 22-minute
per-cycle training budget (`--max-minutes`, which only checks *between* epochs) could be
silently blown by a single overlong epoch, since the orchestrator's `subprocess.run()` calls
have no external kill-timeout. Resolved by computing the actual numbers rather than guessing:
phase-2 cycles train on a much smaller *delta* dataset (~1.24M new positions per 25-minute
window at ~826.5/sec combined, vs. phase 1's full 3.84M), and epoch time scales roughly
linearly with dataset size — so a phase-2 cycle epoch is expected at ~7 minutes, comfortably
fitting 3 epochs inside the 22-minute cap. No architecture change was needed.

**Status at time of writing**: Phase 1 complete — 2 epochs run (epoch 1: val_mae=0.3889,
val_corr=0.6510; epoch 2: val_mae=0.3795, val_corr=0.6592, both improving) before the 40-minute
wall-clock cap ended it; best checkpoint (epoch 2) saved to `training/checkpoints/phase1.pt`.
Phase 2 launched immediately after and confirmed healthy: all 3 self-play+depth-4-grading
workers producing data (~19K positions each within the first few minutes, on pace with the
measured ~826.5/sec combined rate). First phase-2 fine-tune cycle expected around the 25-minute
mark. Once all 4 phases complete, planned verification (not yet run): export via
`training/model/export.py` to `nnue_weights.bin`, `correlation_eval.py`/`tactical_sanity.py` for
eval sanity, and re-run the §16/§17 depth-at-time comparison on both reference positions to see
whether this training approach (shallow-teacher + synthetic self-play + real-game fine-tune)
reduces the search instability (PV/score oscillation across iterations) that §17 diagnosed as
the current bottleneck on the hard position. This section will be updated as each phase
completes.

**Phase-2 cycle 1 confirmed working** (fired at the ~25min mark as expected): 1,094,235 new
self-play+depth-4-graded positions accumulated (close to the ~1.24M/25min estimate above),
dataset built (1,039,524 train / 54,711 val), warm-start from `phase1.pt` (epoch 2,
val_mae=0.3795) confirmed loading correctly, training started at 03:06:10. No epoch complete
yet as of this check, but CPU accounting confirms it's actively computing (573+ CPU-seconds
and climbing), consistent with the ~5.9min/epoch expected for a 1.04M-row dataset at this
scale (matches the ~7min/epoch estimate from earlier in this section). Self-play generation
remains healthy through this check: ~434-436K positions per worker after ~31 minutes of phase
2 runtime (~233/sec/worker, in the same range as the ~275/sec/worker measured earlier), and
all 3 worker processes confirmed actively using CPU (~1,426 CPU-seconds each over that span),
not hung. No failures observed.

**Cycle 1 finished**: epoch 1 val_mae=0.4583/val_corr=0.8562, epoch 2 (best, saved)
val_mae=0.4409/val_corr=0.8621 — both epochs improved within the cycle, confirming the
warm-start fine-tune loop is stable (no divergence/NaN). Note this val_corr is not directly
comparable to phase 1's (0.66) since it's measured against a different held-out split (this
cycle's self-play/depth-4 data, not phase 1's original corpus) — not claimed as a quality
jump, just reported as-is. Cycle 1 took ~27min total (dataset build + 2 epochs of training)
against a nominal 25min slot, so subsequent cycles land roughly every ~50-55min (25min sleep +
~27min work) rather than exactly every 25 — expected given the loop design (the sleep interval
doesn't account for the previous cycle's work time), not a problem. Self-play generation
remains healthy: ~886-888K positions/worker by the ~52min mark of phase 2 (~203/sec/worker,
consistent with earlier measurements), all 3 workers confirmed actively computing via climbing
CPU time. No failures.

**Cycle 2 in progress**: triggered at 03:57:11 with 1,955,845 new positions — nearly double
cycle 1's delta, because cycle *inter-arrival time compounds*: the outer loop always sleeps a
fresh 25min after whatever the previous cycle's build+train work took, so the gap is
25min + prior-cycle-work-time, not a fixed 25min. Cycle 1 took ~27min of work → ~52min gap
before cycle 2 (matches observed). Warm-started correctly from cycle 1 (val_mae=0.4409); epoch
1 result: val_mae=0.4273/val_corr=0.8591 (improving trend continues), and epoch 1 alone took
1267.8s (~21.1min), already close to the cycle's 22min cap, so epoch 2 is expected to push
total training time to ~42-45min. This means cycle 3's gap will stretch further still
(≈25+45≈70min) — cycles are getting fewer and larger as the run progresses. This is an
accepted, reasoned-through consequence of the loop design (not a bug): the growing per-cycle
dataset and compounding gap mean phase 2 will complete somewhat fewer, larger fine-tune cycles
than the nominal ~7 originally estimated for a 180min budget, but each one is still a valid
warm-started fine-tune step. Process health reconfirmed via climbing CPU across all 3 self-play
workers (~4,659-4,671 CPU-sec, up ~1,690s over the prior ~40min check-in) and the active
train.py process. No failures. `relabel.log`'s visible content is stale (leftover from the
earlier killed depth-2 run, ending ~4.26M/6.12M) — phase 3 has correctly not started yet, since
phase 2's 180min budget (started 02:39:48) still has time remaining.

**Cycle 2 finished**: epoch 1 val_mae=0.4273/val_corr=0.8591, epoch 2 (best, saved)
val_mae=0.4189/val_corr=0.8610 — both epochs improved, cycle took 44.35min total (trigger
03:57:11 → checkpoint saved 04:41:32), confirming the compounding-gap prediction.

**Cycle 3 in progress**: triggered 05:06:37 with 2,558,678 new positions — matches the
predicted ~69.35min window (cycle 2's 44.35min work + 25min sleep) almost exactly: 2.56M
positions / 69.35min ≈ 615/s combined ≈ 205/s/worker, consistent with the established rate.
Dataset: 2,430,745 train / 127,933 val, noticeably larger than cycle 2's. Given the scaling
(epoch time roughly proportional to train-row count), cycle 3's first epoch is expected to
take ~27-28min — which would exceed the 22min cap on epoch 1 alone, so cycle 3 will likely
save after just 1 epoch rather than 2 (still a valid checkpoint, since a fresh warm-start's
first epoch is always "best" by definition; not a failure). This means cycle 3's completion is
expected to land right around phase 2's nominal 05:39:48 deadline — the next check specifically
watches for the phase 2 → phase 3 transition. Process health reconfirmed: all 3 self-play
workers climbing steadily (~6,734-6,751 CPU-sec, +~2,080s over the prior 40min check-in ≈ 0.86
cores each), cycle 3's training process actively computing (939 CPU-sec and rising as of this
check). No failures.

**Phase 2 complete.** Cycle 3 finished exactly as predicted: 1 epoch only (val_mae=0.4141,
val_corr=0.8643, 26.65min), saved at 05:36:52 — just under the deadline-minus-60s cutoff, so
the outer loop allowed one more cycle to start. **Cycle 4** triggered at 05:39:50 (1,137,959
new positions, smaller delta since cycle 3's window was shorter), completed all 3 epochs this
time (dataset was back down to ~1.08M rows, so epochs were fast at ~6.2min each): epoch 1
val_mae=0.4153/0.8648, epoch 2 val_mae=0.4099/0.8665, epoch 3 not separately confirmed in the
log excerpt but completed without hitting the time/patience caps, checkpoint saved to
`training/checkpoints/phase2_cycle4.pt`. **Phase 2 total: 4 fine-tune cycles over ~3h19min
wall-clock** (02:39:48 → 05:59:15 — a bit over the nominal 180min due to the compounding
cycle-gap effect observed throughout this section, plus that one extra cycle admitted right at
the deadline boundary). Self-play workers confirmed terminated cleanly at wrap-up (no
processes with those start-times remain). val_mae/val_corr trend across all 4 cycles (not
strictly comparable across cycles' differing held-out splits, but directionally consistent):
0.4409/0.86 → 0.4189/0.86 → 0.4141/0.86 → 0.4099/0.87 — steadily improving, no divergence or
instability observed across the whole phase.

## 21. Phase 3: regrading the full 6.12M base corpus at depth=4

Started immediately at 05:59:15 (`training/data/relabel_shallow.py --depth 4 --workers 6`),
writing to `training/data/raw/shallow4_labeled.jsonl`. Progress at first check: 300,000/
6,122,607 positions done, **~1,450/sec sustained, ETA ~67min** — a bit faster than the ~1,215/
sec estimated earlier in this log (scaled from the depth-2/depth-4 single-worker ratio), likely
because 6 real cores with no self-play contention (phase 2's workers are now stopped) gives
this phase the full machine rather than sharing it. This regrade output supersedes the earlier
partial depth-2 file (`shallow2_labeled.jsonl`, used only as phase 1's seed data) — phase 4 will
train on this fresh, fully depth-4-consistent 6.12M corpus instead.

Progress check at 06:40: 3,460,000/6,122,607 done (56.5%), rate holding stable at ~1,425/sec,
ETA ~31min (on track to finish ~07:11-07:12). Process health confirmed via the expected
per-chunk worker churn (6 fresh worker processes spawned for the current 20K-row chunk, plus
the persistent parent coordinator process climbing steadily) — matches the chunked
multiprocessing design in `stockfish_teacher.py`/`relabel_shallow.py`, not a hang.

**Phase 3 complete**: finished at 07:11:57, took 72.4min total for all 6,122,607 positions —
close to the ETA estimates throughout. `training/data/raw/shallow4_labeled.jsonl` is the
final output, ready for phase 4.

## 22. Phase 4: final fine-tune on the depth-4-regraded 6.12M corpus

Started immediately at 07:11:57. Dataset build: **5,511,584 train / 290,083 val** (took
~7.3min, 07:11:57 → 07:19:13 — close to the ~6.4min scaled estimate from phase 1's 4.04M/
4.2min build). Training started 07:19:13, warm-started from `phase2_cycle4.pt` — confirms
phase 2's cycle 4 did complete a 3rd epoch as suspected (val_mae improved further to 0.4051
from the previously-logged epoch 2 value of 0.4099). Config: hidden=1024, lr=5e-4 (lower than
phase 1/2's initial 2e-3, appropriate for a final polishing fine-tune), epochs capped at 100
with early-stop-patience=4 and a 60min wall-clock ceiling, output to
`training/checkpoints/final_model.pt`. Per the phase-1 timing baseline (1300s/epoch on 3.84M
rows), a single epoch on this larger 5.51M-row dataset is expected to take roughly
1300×(5.51/3.84) ≈ 1865s (~31min) — so this run will likely complete only 1-2 epochs within
its 60min cap, similar to phase 1's pattern. Process health confirmed at the 11.6min mark:
actively computing (3,350 CPU-seconds accumulated, ~4.8x core utilization, consistent with
phase 1's parallelism profile), not hung. No epoch result yet as of this check.

**Epoch 1 result**: val_mae=0.4096, val_corr=0.6930, took 1,819.6s (~30.3min) — matches the
~31min estimate closely. Saved as best (first epoch of a fresh run always is; not directly
comparable to phase 2's self-play val_mae since this is a different held-out split — real-game
positions, not synthetic self-play). As of the 48.7min mark, training is into epoch 2 (no
second result logged yet), confirmed actively computing (13,797.7 CPU-seconds, ~4.7x core
utilization, consistent with the established parallelism profile). Given the 60min cap and
~30min/epoch, epoch 2 is expected to push total elapsed to ~60min before the wall-clock check
stops the run after that epoch completes — same pattern as phase 1's 2-epoch finish. Expected
to wrap up around 08:19-08:20.

**Phase 4 complete.** Epoch 2: val_mae=0.4051, val_corr=0.6975 (best, saved) — improved over
epoch 1's 0.4096/0.6930. Stopped by the 60min wall-clock cap immediately after epoch 2, exactly
as predicted. Final checkpoint: `training/checkpoints/final_model.pt`.

## 23. Full 4-phase pipeline complete (6.47h total) — summary and next steps

`training/checkpoints/LATEST_4PHASE.txt` confirms the entire pipeline finished at 08:20:14,
**6.47 hours** after it started at 01:51:59, pointing to `training/checkpoints/final_model.pt`
as the winning checkpoint. Recap of the full journey (per the user's explicit 4-phase
instruction, replacing all earlier plans in this session):

- **Phase 1** (~48min): seeded the network (hidden=1024, "Stockfish-sized" per the user's
  earlier request) on the 4,042,063 positions already depth-2-graded from an earlier,
  interrupted relabeling run — reused rather than discarded. 2 epochs (40min cap):
  val_mae 0.3889→0.3795, val_corr 0.6510→0.6592.
- **Phase 2** (~3h19min): continuous Stockfish self-play generation (3 workers, the schedule
  refined in §19: forced-random opening, depth-7 settle, depth-2-with-sparse-random
  middlegame, depth-7 endgame) with a *separate* depth-4 grading pass on every visited
  position, incrementally fine-tuning the running checkpoint every cycle (4 cycles total, each
  warm-started from the last). val_mae/val_corr progression: 0.4409/0.86 → 0.4189/0.86 →
  0.4141/0.86 → 0.4051/0.87 (cycle-over-cycle splits aren't strictly comparable to each other,
  but the trend held steady with no divergence across ~5.4M total self-play positions
  generated).
- **Phase 3** (72.4min): regraded the *entire* 6,122,607-position base corpus (real Lichess
  games + curriculum positions) at depth=4 from scratch, superseding the partial depth-2 file
  phase 1 used — ensures phase 4 trains on a fully depth-4-consistent, real-game corpus.
- **Phase 4** (~61min): final fine-tune of the phase-2 model on that depth-4-regraded 6.12M
  corpus, lr=5e-4 (lower, for polishing rather than aggressive relearning). 2 epochs (60min
  cap): val_mae 0.4096→0.4051, val_corr 0.6930→0.6975.

**Two real bugs were found and fixed along the way** (not just training-plan iteration):
`stockfish_teacher.py`'s worker-result queue ordering bug (§18, could silently hang the whole
pipeline under engine-process oversubscription) and the Windows-vs-bash `/tmp` path mismatch
in the orchestrator (§18, would have made `wait_for_relabel()` poll forever). Both are why the
pipeline actually ran unattended for 6.5 hours without getting stuck.

## 24. Verification: export + correlation/tactical sanity + depth-at-time re-test

`nnue_weights.bin` backed up (`nnue_weights_pre_4phase_backup.bin`) before overwriting, then
`final_model.pt` exported via `training/model/export.py` (hidden=1024, 8 buckets, confirmed
loading correctly — `network.h`/`.cpp`/`accumulator.*` are fully dynamic on hidden size, no
hardcoded 384 anywhere, so no engine code changes were needed). Full binary rebuild clean.
Quick sanity: engine still produces legal moves and still finds the `Qd1+` queen-sac tactic
from earlier sessions.

**Correlation against real-game positions** (`correlation_eval.py` on 1,000 sampled positions
from the fresh depth-4-regraded `shallow4_labeled.jsonl`, since the script's default
`lichess_400k.jsonl` reference file doesn't exist in this repo):

| | old (pre-session) weights | new (`final_model.pt`) |
|---|---|---|
| pearson r | 0.7275 | **0.8357** |
| MAE (cp) | 299.1 | **106.5** |

A real, substantial improvement in how well the eval tracks a deep-search-quality reference.

**Tactical sanity** (`tactical_sanity.py`) — **regressed, 6/7 → 5/7**:

| case | old cp (pass?) | new cp (pass?) | threshold |
|---|---|---|---|
| white up queen, bare kings | 2374.8 ✓ | 615.6 ✓ | >500 |
| black up queen, bare kings | -1204.0 ✓ | **-484.2 ✗** | <-500 |
| white up queen, middlegame | 902.6 ✓ | 761.1 ✓ | >400 |
| black up rook, middlegame | -251.9 ✗ (soft-fail) | **-193.8 ✗ (worse)** | <-300 |

Eval magnitudes are dramatically compressed across the board (e.g. white-up-a-queen-bare-kings
2374.8→615.6cp) — compressed enough that one previously-solid case now fails outright, and the
pre-existing rook-deficit soft-fail got measurably worse. Most likely cause: a depth-2/4
Stockfish search on a decisively winning position doesn't fully explore the winning
continuation the way a deep search does, so the teacher's own reported cp for those positions
is itself more conservative than a deep-search teacher's would be — and the network faithfully
learned that compression. This is a real, direct downside of the shallow-teacher hypothesis
this session set out to test, not a training bug.

**The actual target test — depth-at-time vs. the §16/§17 baseline** (same two reference
positions, same methodology):

*Easy position*: node count got **worse** at matching depths (depth 12: 152,410→382,444 nodes;
depth 14: 311,015→1,036,273 nodes) — a real regression here, not an improvement.

*Hard position* (the one §17 specifically diagnosed as unstable): mixed at shallow depths
(depth 11: 360,585→659,033 nodes, worse), but **depth 14 node count nearly halved**
(1,896,090→1,096,824). More tellingly, the **score sequence across iterations** — the direct
signature of the PV instability §17 diagnosed — changed shape: old model oscillated
(89→80→83→84→86, dropping then climbing across depths 10-14), new model **converges smoothly
and monotonically** (58→63→70→70→70, flattening out). This is a genuine, direct sign that
search instability improved on exactly the position it was diagnosed on.

**Honest overall verdict: a real, mixed result, not a clean win.**
- ✅ Correlation/MAE against real-game positions improved substantially.
- ✅ Search instability on the hard/tactical position genuinely improved (smoother score
  convergence, ~42% fewer nodes at depth 14).
- ❌ Tactical sanity regressed (6/7→5/7) — shallow-teacher eval-magnitude compression is a real
  cost, not a training bug.
- ❌ Easy/quiet-position node efficiency got worse, not better — this session's training
  approach did not uniformly help pruning efficiency.

This session's four-phase training pipeline is complete and verified both ways (what improved,
what didn't) — decision on next steps (e.g. deeper-teacher rebalancing to fix the magnitude
compression, more pruning work now that this eval is in place, or further training iteration)
is left to the user.

## 25. Closing the feature gap vs. real Stockfish: opponent-king + threat features

Following the deep-dive comparison against Stockfish's actual NNUE architecture (HalfKAv2_hm +
Full_Threats), two structural gaps were identified and implemented:

1. **Opponent king as an explicit feature** (closing the HalfKP→HalfKA gap). Previously kings
   were entirely excluded from feature indexing except as the perspective anchor. Now the
   opponent's king occupies a dedicated 11th piece-type slot (own king is still never a
   feature — it only ever anchors the king bucket). This grows the placement feature space
   from 20,480 to 22,528 (`32 king buckets × 11 piece-type-slots × 64 squares`).

2. **Threat features** ("who attacks whom") — a deliberately compact approximation of
   Stockfish's Full_Threats, not a replica of its exact 60,720-dimension, square-level scheme
   (which even Stockfish's own 17.1 release reportedly didn't activate). Ours uses type-pair
   granularity: `(king bucket, own-attacks-enemy | enemy-attacks-own, attacker piece type
   [p/n/b/r/q/k], victim piece type [p/n/b/r/q/k])`, deduped per unique triple — 2,304
   additional feature slots (`32 × 2 × 6 × 6`). Total feature space: **24,832**
   (`training/model/net.py NUM_FEATURES`, `engine/human_limit/nnue_features.h kNumFeatures`).

**Verification.** Both features implemented twice independently (Python `encode.py` using
python-chess for attack computation; C++ `nnue_features.cpp` using hand-rolled geometric ray
casting on `chess::BoardArray`, since the runtime hot path needs this without a python-chess
equivalent). All 42 golden fixtures (`gen_golden.py`/`golden_features.json`,
`test_nnue_features.cpp`) match exactly between the two implementations. The pre-existing
incremental accumulator tests (`test_accumulator`, `test_accumulator_bb`) still pass unchanged
— threat features are deliberately kept out of the incremental diff path entirely (see below),
so the already-verified placement-feature diffing logic was not touched.

**The speed risk, measured and mitigated.** Threat features are not incrementally diffable the
way placement features are — a single move can change attack relationships across the whole
board (discovered attacks, opened/closed lines), unlike a placement change which touches O(1)
squares. Given this project's explicit priority on not regressing the hard-won 5x eval-speed
work, the design deliberately isolates the risk: the existing incremental placement accumulator
(`accumulator.cpp`) is completely untouched; threat features are instead computed fresh from
the board only at evaluation time, in a new `Network::evaluateFromAccumulatorsWithThreats()`
that adds threat-feature embedding rows on top of the (still incrementally maintained)
placement accumulator before the head-forward pass.

Measured directly (same node counts before/after at every depth, isolating pure per-node eval
cost, single fixed reference position): the first, naive implementation (scanning all 64×64
square pairs per perspective, computing attack facts twice — once per perspective, redundantly)
cost **+117% per-node time (2.17x slower)** — depth 12 went 173ms → 376ms. This was optimized
down to **+36% (1.36x)** — 173ms → 247ms — via two fixes: (1) iterating only over actually
occupied squares (collected once into a small fixed array) instead of blindly scanning all 64
origin × 64 target square pairs regardless of occupancy, and (2) computing the attack-fact set
**once** per evaluation (attack geometry is perspective-independent) and deriving both
perspectives' feature-index lists from that single shared computation, instead of redoing the
full O(pieces²) attack enumeration twice.

**Status: 36% per-node overhead accepted as a real, honestly-measured cost**, not hidden or
rounded away — the trade is that the network now receives direct opponent-king and tactical-
threat signal it previously had to infer purely from placement patterns. Whether the eval-
quality gain is worth the speed cost can only be judged empirically (depth-at-time vs.
Stockfish, plus game/tactical-sanity results) once a model is actually trained on the new
feature space — old checkpoints are incompatible (embedding table shape changed 20,480→22,528
rows plus the new 2,304 threat rows = 24,832 total, so `--init-from` warm-starting is not
possible; a fresh training run is required).

**Retraining.** Dataset build (`build_dataset.py` on the existing depth-4-regraded 6.12M corpus
from §21, re-encoded with the new feature scheme, 5,801,667 usable rows) took ~46 minutes
against a ~20min small-sample-benchmark projection, due to heavy system-wide resource
contention — five concurrent VS Code processes (including a C++ IntelliSense indexer,
`cpptools`, likely triggered by this session's C++ edits), four Chrome processes, Windows
Defender, and telemetry services were all found competing for the same CPU cores and pushing
free system memory down to ~2-3% at the worst point (recovered to ~2.2GB free once the build
finished) — not a problem with the encoding approach itself. Training then ran fresh from
scratch (no warm-start possible — the embedding table shape changed 20,480→24,832 rows), 2
epochs before hitting the 50-minute cap: epoch 1 val_mae=0.4324/val_corr=0.6659 (1,799.5s),
epoch 2 (best, saved) val_mae=0.4168/val_corr=0.6560 (1,819.9s) →
`training/checkpoints/threats_model.pt`.

**Full regrade**, exported to `engine/human_limit/nnue_weights.bin` (prior weights backed up to
`nnue_weights_pre_threats_backup.bin`), all tools rebuilt, sanity-checked (legal moves, no
crash, `Qd1+` queen-sac tactic still found):

*Correlation* (`correlation_eval.py`, 1000 samples from `shallow4_labeled.jsonl`): **r=0.8291,
MAE=111.4cp** — essentially unchanged from the prior 4-phase model (r=0.8357, MAE=106.5cp), a
tiny regression within noise. The new features didn't move this metric.

*Tactical sanity*: **still 5/7**, same two failing cases as §24, magnitudes essentially
unchanged (black-up-queen-bare-kings: -484.2→-486.2cp against a <-500 threshold;
black-up-rook-middlegame: -193.8→-158.6cp, further from the <-300 threshold, if anything
slightly worse). The new features did not fix the material-deficit-magnitude-compression issue
diagnosed in §24 as a likely consequence of shallow-teacher training — makes sense, since that
compression is about eval *magnitude* calibration, not tactical *awareness*, and threat
features address the latter.

*Depth-at-time* (same methodology and reference positions as §16/17/24, 30s budget): **the
real, coherent finding of this section.**

| position | depth | before threats | after threats | wall-clock delta |
|---|---|---|---|---|
| easy (quiet) | 12 | 4,303ms / 382,444 nodes / score 41 | 6,239ms / 368,027 nodes / score 32 | **45% slower** |
| easy (quiet) | 14 | 11,656ms / 1,036,273 nodes / score 26 | 17,653ms / 1,046,909 nodes / score 20 | **51% slower** |
| hard (tactical) | 12 | 8,639ms / 793,109 nodes / score 70 | 5,052ms / 310,935 nodes / score 67 | **41% faster** |
| hard (tactical) | 14 | 11,925ms / 1,096,824 nodes / score 70 | 11,574ms / 713,991 nodes / score 77 | **~3% faster** |

On the quiet position, node counts are essentially unchanged but the 36% per-node overhead
(§25 above) is pure cost with nothing to offset it — a quiet position has few real
attacker/victim relationships for the threat features to usefully encode. On the tactical
position, node count dropped **2.55x at depth 12** and **1.54x at depth 14** — the richer eval
improved move ordering/pruning enough to more than pay for the per-node cost, net **faster**
wall-clock despite doing more work per node.

**Verdict: a real, position-dependent trade, not a uniform win or loss.** Threat/opponent-king
features cost real speed on quiet positions and buy real search efficiency on tactical ones —
exactly the shape expected from features that explicitly encode attack/defense relationships.
Since actual games are a mix of both quiet and tactical phases (and the tactical phases are
typically where accuracy matters most, e.g. avoiding blunders/finding combinations), this is
plausibly a net positive for real play despite hurting raw depth on quiet positions — but that
can only be confirmed by actual game-strength testing (vs. old weights / vs. Stockfish), not
these synthetic single-position benchmarks alone. Correlation and tactical-sanity are flat
either way, so the case for keeping this change rests specifically on the tactical-position
search-efficiency result, not on eval-quality metrics. Old weights are preserved
(`nnue_weights_pre_threats_backup.bin`, `nnue_weights_pre_4phase_backup.bin`) for easy
comparison or rollback.

## 26. The "fusion" experiment: isolating pruning quality from eval quality, then removing it

**Motivation.** After §25's NNUE work, the engine was still losing ~50% of games to
Stockfish@1800 and clearly weaker than @2000. The open question: is the bottleneck our search
(alpha-beta/pruning quality) or our eval (the NNUE itself)? Rather than guess, we built a
diagnostic: keep our own search/pruning code entirely intact, but replace the NNUE eval call
with a live query to a real Stockfish process at a shallow depth (1-2), via
`ExternalEvalFn`/`StockfishEvalBridge` (Windows-pipe IPC to a persistent `stockfish.exe`,
`tools/sf_eval_bridge.h/.cpp`) and a `human_sfeval` mode in `bestmove_cli.cpp`. If this
"fusion" engine played *well* despite a much worse node budget (each node now costs an IPC
round-trip, ~200-300us vs. NNUE's ~17us), that would mean our pruning is fine and eval quality
is the real bottleneck — the reverse would mean pruning itself needs work.

**Result: fusion played very well.** With search/pruning improvements layered on top (see
below), the fusion engine scored 83.3% vs. both Stockfish@1800 and @2000 in initial batches,
and remained competitive (~50%) even at @2200 — all while running at ~10-13x fewer nodes/sec
than the NNUE path. This is strong evidence the search/pruning framework is not the limiting
factor; the NNUE eval quality is.

**What we found and fixed along the way, via a batch-then-ablate methodology** (implement
several changes at once, measure net effect on a fast proxy metric — average centipawn loss
vs. Stockfish ground truth on a fixed 20-position sample from `training/eval/
move_quality_suite.py` — then disable one change at a time to attribute gains/losses, since
testing changes one-by-one against full games is far too slow to iterate on):
- **Two changes were kept permanently** (validated as net-positive independent of eval
  source, so retained in the real `human` engine after fusion removal): **check extensions**
  (+1 ply for non-losing checking moves — a standard technique that was simply missing) and
  **quiescence delta pruning** (skip a capture if even winning it outright plus a margin can't
  reach alpha).
- **Two changes were found net-negative and reverted**: an eval-result cache and lowered
  IID/singular/probcut depth thresholds. Both let the search reach one ply deeper within the
  same time budget — but with Stockfish-depth-1 as the eval oracle, going deeper sometimes
  *hurt* rather than helped. Diagnosed via research (not guessing) as **minimax search
  pathology**: minimax back-up amplifies noise when an evaluator has weak parent/child
  correlation, and a 1-ply Stockfish search is noisy enough (observed score swings like
  254→264→−44→6→40 across depths on the same node) to trigger this. Confirmed concretely: one
  specific position's best move flipped from the correct `c2g6` to a blunder `c2d2` only when
  the search reached one ply deeper via these two changes.
- **Move-stability voting** was added to directly counter that pathology (if the deepest
  completed iteration's move disagrees with a choice stable for the two iterations before it,
  trust the stable one) — measurably helped at `sfeval_depth=2` (11.7 avg centipawn loss vs.
  19.9 without it) but not at depth=1 (too few iterations for "stable" to mean anything).
- **Switched the internal oracle from depth=1 to depth=2** per the same pathology diagnosis
  (more search-per-query means better parent/child correlation, less horizon-effect noise),
  despite costing more time per query.
- **A promising-looking speed idea that didn't pan out**: sending the oracle `position fen
  ROOT moves m1 m2 ...` instead of a fresh arbitrary FEN each query, so it could incrementally
  apply moves (and NNUE-accumulator-update) instead of fully reparsing from scratch. A naive
  single-shot A/B benchmark showed ~37% faster; a rigorous *interleaved* re-test (to control
  for system-noise drift) showed only ~10%, and it produced **zero** measurable difference in
  the actual search's node counts (real move-paths from root are short at our reachable depth,
  so the incremental savings were negligible in practice). Reverted rather than keep unproven
  complexity — exactly the kind of speculative-but-unconfirmed change the batch-then-ablate
  discipline exists to catch.
- Verified real Stockfish-vs-us architecture facts along the way (not assumptions): Stockfish's
  own "Big Network" also uses a 1024-dim hidden layer (same size class as ours) — the actual
  speed gap is int8/int16 quantized SIMD (`_mm256_add_dpbusd_epi32`) vs. our plain float32, plus
  Stockfish's two-tier cheap/expensive network selection (a 128-dim "Small Network" for simple
  positions) that we don't have at all.

**Removal.** Once the diagnostic question was answered, the fusion machinery was deleted
entirely per instruction: `tools/sf_eval_bridge.h/.cpp` removed, `human_sfeval` mode removed
from `bestmove_cli.cpp`, `ExternalEvalFn`/eval-cache/lowered-thresholds/move-stability-voting
removed from `search.h/.cpp`. Check extensions and quiescence delta pruning were kept
unconditionally (no longer behind toggles) since they're generic, validated improvements, not
tied to the noisy-external-eval scenario. Confirmed via full test suite + a depth/speed spot
check that the real `human` engine's behavior is sane post-cleanup (same best move on the
standing tactical reference position as the pre-experiment baseline, node/depth counts in the
same ballpark). Next: NNUE architecture improvements (the actual bottleneck this experiment
identified), not further search tuning.

## 27. int8/AVX2 quantization of the NNUE head — a real but modest win, then stopping search-speed work

Given external research on how Stockfish gets its NNUE speed (int8/int16 quantized SIMD,
`_mm256_add_dpbusd_epi32`/VNNI where available, dual small/big network selection), attempted to
apply the applicable parts to our own NNUE (the IPC/oracle-cost parts of that research described
the already-removed §26 fusion experiment, not our current in-process NNUE, and didn't apply).

**What was implemented.** Quantized just the `fc1` layer (32x2048 per bucket, ~98% of
head-forward's multiply-adds) to int8 weights, keeping `fc2`/`fc3` in float32 (negligible cost
either way) and leaving the accumulator/incremental-update path untouched (already fast,
heavily tested at float precision). This CPU has AVX2 but no VNNI, so used the standard
non-VNNI technique: `_mm256_maddubs_epi16` (requires one unsigned operand) +
`_mm256_madd_epi16` (widens the int16 partial pair-sums to int32 before they can overflow).
Since the accumulator input has no natural non-negative range (no ReLU before fc1 in this
architecture, unlike Stockfish's clipped-accumulator design), quantized it to a signed
[-63,63] range and shifted to unsigned [0,126] for the unsigned operand, correcting for the
shift via a precomputed per-output-neuron row-sum of the quantized weights
(`dot(w,x) = dot(w,ux) - 63*sum(w)`, where `ux = x+63`). Kept the original float32 `fc1w_` and
a `headForwardFloatReference` method as a permanent ground-truth check, not part of the hot path.

**A first attempt (portable, no intrinsics) made things 4x slower, not faster** — direct
confirmation of what the research warned: auto-vectorizers don't reliably turn scalar
int8-with-explicit-casts loops into real SIMD, so "just switch the types to int8" without
hand-written intrinsics is actively counterproductive. Only the explicit AVX2 intrinsics version
delivered a real win.

**Accuracy validated across 2000 real positions** (vs. the float32 reference): mean absolute
difference 0.70cp, max 8.05cp, correlation 0.999992, zero sign flips. Quantization is not a
meaningful accuracy cost.

**Speed measurement was initially confounded by heavy background system contention**
(cpptools C++ IntelliSense indexing + multiple VSCode/Chrome processes — the same class of
interference diagnosed earlier this session, ~22% free memory at the worst point) that inflated
*all* absolute timings 3-4x, briefly making the quantized path look slower than the original.
Fixed by measuring quantized-vs-float ratio via an *interleaved* same-process comparison
(alternating small batches of each within one run, so both experience identical contention) —
this gave a stable, reproducible **2.7-2.8x speedup on the head-forward computation alone**
across three repeated runs.

**Honest net assessment: the speedup on the whole per-node eval is real but modest, not
dramatic.** The accumulator recompute/incremental-update path (untouched, ~7.8us/~1.1us
respectively) is now comparable to or larger than the quantized head-forward's cost, so
quantizing only fc1 doesn't proportionally shrink total per-node eval time the way the 2.7x
head-forward number alone suggests — call it roughly 1.5x on the full eval, not 2.7x. A clean
"more search depth in the same wall-clock budget" demonstration wasn't obtained before running
out of good measurement conditions (the search-level benchmark, unlike the interleaved
comparison, is directly exposed to whatever contention exists at the moment it's run).

**Decision: stop chasing further search/eval micro-speedups here.** Between this and §26's
finding (a search running at 10-13x fewer nodes/sec than our real NNUE still played at 83%+
against 1800-2000 rated Stockfish), the evidence points the same direction twice now: search
speed and pruning quality are not where the remaining Elo is. Moving to NNUE architecture and
training-quality improvements as the next phase.

## 28. Push to 2500: baseline placement, persistent UCI + Lazy SMP, and a failed eval experiment

New explicit goal: reach 2500 Elo, attacking BOTH sides (NNUE eval accuracy AND search depth),
iterating in batches without getting stuck in testing (keep making progress while matches run).
Elo is gauged by playing the locally-downloaded Stockfish at calibrated `UCI_LimitStrength`/`UCI_Elo`
levels 1500..3000.

**Baseline placement (current threats weights, process-per-move `bestmove_cli`, 500ms/move, 6 games/level):**
1500: 91.7% · 1800: 25.0% · 2100: 66.7% · 2400: 16.7%. The 1800/2100 inversion is small-sample
noise (6 games can't resolve <150 Elo); honest read is **~1950-2100 Elo now** — crushes 1500,
roughly even 1800-2100, loses to 2400. Target 2500 is ~400-600 Elo away.

**Persistent UCI engine (`tools/uci_engine.cpp`, new).** Every move in the old harness cold-started
a fresh `bestmove_cli` process, reloading the ~100MB NNUE and discarding the transposition table
between moves. Built a real minimal-UCI front-end that keeps one `Searcher` alive for the whole game,
so TT / history / correction-history persist across moves (a free strength gain) and weights load
once. Speaks `uci/isready/ucinewgame/position/go (movetime|wtime/btime)/setoption/quit`, drivable by
python-chess and any GUI. New match harness `tools/match_uci.py` (vs Stockfish) and `tools/ab_weights.py`
(head-to-head between two of our own weight files / thread counts, for low-variance eval A/B). Weights
path is overridable via `HL_WEIGHTS` env so two configs can play each other.

**Lazy SMP multi-threaded search (the 6 idle cores).** Refactored `Searcher` to share one
transposition table (`SharedTT`, `std::shared_ptr`) across N worker threads; helper threads search
their own position copy with a Stockfish-style depth-skipping diversity table so they race ahead and
seed the shared TT for the main thread. Default is 1 thread (behaviour byte-identical to before —
single-thread node counts unchanged). `setThreads` wired to the UCI `Threads` option.

Three real bugs found and fixed during SMP bring-up, each verified by the crash disappearing:
1. **Position data race** — helpers did `Position copy = pos` *inside* the thread while the main
   thread was already mutating `pos` (its `std::map` repetition table) via make/unmake → crash.
   Fixed by snapshotting each helper's position copy synchronously in the main thread before any
   search starts.
2. **mingw `thread_local std::vector` concurrency crash** — the eval's scratch buffers
   (`network.cpp`: the head-forward `x`/`uxq`, and the threat-augmented accumulators) were
   `thread_local std::vector`s. Under std::thread helpers on this MSYS2 mingw build they segfaulted
   intermittently even on a 50ms search (a known mingw TLS-with-nontrivial-ctor hazard). Confirmed by
   bisection: a no-op helper never crashed, a helper running the real eval crashed ~50% of runs, and
   switching those buffers to plain stack locals made 8/8 runs clean. Kept stack-local (the per-call
   allocation is negligible against the threat computation; single-thread node counts are identical).
3. **Ownership-ordering hazard** in the (guarded) spawn loop — objects are now moved into their owning
   vectors *before* the thread is spawned, so a thread-creation throw can never free an object a
   running helper still references. Helper bodies and the whole spawn loop are wrapped so resource
   pressure degrades gracefully to fewer threads / single-thread instead of taking down the process.

An important environment note surfaced here: the earlier SMP crashes were reproducible **only while
the training run was consuming ~5-6 cores and RAM was at ~22% free** — a failed page commit under
memory pressure surfaces as an uncatchable access violation. With the machine free (6GB+ free), SMP
runs clean. **Measured depth-at-time (machine free, 3s budget):** 1 thread reaches depth 11 on both
the standard easy and hard reference positions; **6 threads reaches depth 12 (easy) / 13 (hard)** —
a real +1-2 plies. A 6-thread-vs-1-thread game match (same weights) is running to confirm the Elo
value.

**A failed eval experiment, recorded honestly.** Hypothesis: the tactical-sanity magnitude
compression (§24/§25 — "up a queen" scored far below its material value) is costing strength, so
train it out. Implemented a hybrid loss (`train.py --mse-weight`: WDL + an eval-space MSE term) plus
heavy oversampling of the decisive curriculum positions (`--aux-dataset`/`--aux-repeat`,
`dataset.py` CSR-concatenation with a verified offset test), warm-started from the current threats
model. It **worked on the metrics and failed on strength**: val correlation jumped 0.656 → 0.745 and
val MAE improved, but a 30-game head-to-head of the epoch-1 hybrid weights vs the current production
weights (`ab_weights.py`, 400ms) came back **2.0/9 (~22%) — clearly weaker** before it was stopped.
The lesson matches why Stockfish trains pure-WDL: magnitude calibration and decisive-position
oversampling improved a *val metric* that does not track move-ranking quality, and the aggressive
mix distorted the middlegame judgement that actually wins games. Production weights were never touched
(the hybrid net was exported to a separate file). **Takeaway carried forward: judge eval changes by
games, not by val correlation; the principled eval lever is a better/deeper teacher, not loss tricks.**

## 29. Blunder-mining active learning (hard-example) — first iteration WON on games

Per the user's idea: instead of blindly relabeling a generic corpus, target the model's
*actual* weak points. New pipeline `training/data/mine_blunders.py`:
1. Play games (our persistent `uci_engine` vs Stockfish at a calibrated Elo), record every ply.
2. Grade every position with Stockfish (depth 12) and flag positions where OUR move dropped
   our-side eval by >= 150cp — i.e. our engine blundered. **Mined from ALL games, including
   ones we won/drew** (per the user: an unpunished blunder is still a real weak point).
3. Perturb each blunder position into 10 legal near-neighbours (one random piece added / removed /
   relocated on either side) for generalization.
4. Deep-grade (depth 14) the blunders + perturbations → a training jsonl of weak-point positions.

Ran two instances in parallel (vs SF-2200 and vs SF-1900, using otherwise-idle cores): **~400
blunders → 4,389 deep-graded weak-point positions**. The SF-1900 run (games we mostly win) still
produced 194 blunders, directly confirming the "even the wins" intuition.

**Fine-tune** (`training/finetune_blunders.sh`): pure-WDL (no magnitude tricks — §28's lesson),
warm-started from the current weights, on a 1M-row base anchor (`subset_dataset.py`) + the
weak-point set oversampled 30x (~125k), low LR 2e-4, 2 epochs. General-position val MAE even
*improved* (0.4168 → 0.4087), an early sign it sharpened weak points without distorting normal play
(unlike §28's hybrid). **A/B vs the previous production weights (threads=1, isolating the eval):
7.5/8 (~94%) — a decisive, unambiguous win**, the mirror image of §28's failed hybrid (2/9).
Adopted as the new production `nnue_weights.bin` (previous weights backed up to
`nnue_weights_pre_blunderft.bin`). The head-to-head margin overstates the true external Elo gain
(same engine, so the new eval repeatedly exploits the old one's specific blind spot), so a fresh
Stockfish sweep (new eval + SMP) is running to measure the real jump. This validated the
active-learning loop end-to-end; it is now repeatable: mine the new weights' blunders → fine-tune →
repeat.

## 30. Overnight blunder collection (5h), a mate-depth mining artifact caught by direct inspection,
    and active-learning iteration 2 — adopted (68.3% vs iteration 1)

Per direct instruction, launched an unattended 5-hour version of the blunder-mining loop
(`training/data/collect_overnight.py`, new): alternates 4-game batches vs Stockfish@2100 and
Stockfish@1800 (300ms/move), grades every position, flags OUR moves that drop our own eval by
>=150cp (from ALL games, including wins/draws — an unpunished blunder is still a real weak point),
perturbs each into 10 near-neighbours, deep-grades everything (depth 14), and appends immediately
after every cycle so a crash/stop loses at most one in-progress cycle. A running summary json gives
an at-a-glance status. Smoke-tested end-to-end (short budget) before committing to 5 hours unattended.

**Result: 480 games (296W/122D/62L), 1,514 blunders detected, 16,654 deep-graded weak-point
positions** written to `training/data/raw/overnight_blunders.jsonl`.

**A real methodology bug caught by direct inspection of actual examples, not by metrics.** Asked to
show concrete blunder examples (position + Stockfish eval + our own engine's raw eval, via a new
`tools/show_blunder_examples.py`), one flagged "blunder" (`e6d7` in a rook-up endgame) was checked
against a much deeper external analysis (depth 59) and turned out to be **the best available
defense — a forced mate (M7) that could not be avoided by any legal move (the alternatives were
M6)**. Our depth-14 "before" grading hadn't found the forced mate yet (scored it as a merely-large
866cp rather than recognizing it was already lost by force), so any move looked like a "swing" once
a later, deeper-in-the-line score resolved toward the mate value — an artifact of teacher search
depth in already-decided positions, not a genuine mistake. **Fix**: `detect_blunders()` gained a
`before_ceiling` parameter (default 600cp) — positions already this decisive before our move are
skipped entirely, so mining targets genuine turning-point blunders (still-contestable positions that
we made worse) instead of "which flavor of forced loss" noise in dead-lost endgames. Wired into both
`mine_blunders.py` and `collect_overnight.py`. Not applied retroactively to already-collected data
(the position labels themselves are still directionally valid even where the "blunder" framing was
imprecise), but governs all future mining rounds.

**Active-learning iteration 2** (`training/finetune_blunders.sh`, updated to warm-start from the
current best checkpoint and combine all mined weak-point sets including the overnight collection):
combined 21,043 weak-point positions (2,255 + 2,134 from the first mining session + 16,654
overnight), pure-WDL loss, 1M-row base anchor + weak points oversampled 10x, warm-started from the
iteration-1 checkpoint (`blunder_ft.pt`). All 4 epochs improved (val_mae 0.4113→0.4100, val_corr
0.660→0.670) → `training/checkpoints/blunder_ft2.pt`.

**A/B against the iteration-1 production weights** (`tools/ab_weights.py`, 30 games, 500ms,
threads=1): opened rocky (5.5/10, 55% at the 10-game mark — genuinely close to a coin flip, matched
a stated hunch that this iteration might not be better) but the full 30-game sample resolved to
**20.5/30 (68.3%)** — a real, if less dramatic than iteration 1's 94%, improvement. **Adopted** as
the new production `nnue_weights.bin` (previous weights backed up to `nnue_weights_pre_ft2.bin`).
The mine→fine-tune→A/B loop has now been run twice end-to-end, both times judged by games rather
than val metrics, matching the discipline established in §28.

## 31. Eval hot-path speedup: bitboard-native threat features, ~1.9x per-node, real depth-at-time gain

Direct measurement requested of "how many ms to reach each search depth" surfaced a specific,
quantified target rather than a vague slowness complaint. Profiled the actual per-node eval hot path
(`tools/profile_breakdown.cpp`, extended with new sections) used by `Searcher::evalWhiteRelative`:

| component | cost | share of ~6.7-6.9us total |
|---|---|---|
| `pos.toBoardArray()` (bitboard -> mailbox) | 0.24us | 3.5% |
| `computePerspectiveContext()` (x2) | 0.08us | 1% |
| `computeThreatFacts()` | 3.13-3.72us | **~47%** |
| NNUE head forward (the actual neural net math) | 2.83-2.91us | 43% |

`computeThreatFacts()` — the NNUE's "who attacks whom" feature computation added in §25 — was
costing almost as much as the entire neural network forward pass itself, because it ran on a
mailbox (8x8 array) board using geometric ray-walking (O(pieces²) pairwise checks), when the
existing bitboard `Position` already has O(1) attack tables (`attackersTo()`, the same ones `see()`/
`inCheck()` use) that answer "who attacks this square" without a geometric walk. (Pruning-side board
queries — `inCheck()` 0.004us, `see()` 0.022us, `getValidMoves()` ~1.3-1.6us — were already fast and
bitboard-based; the bottleneck was specifically the NNUE input pipeline, not search/pruning
plumbing.)

**Fix, safety-first per this project's established discipline**: wrote `computeThreatFactsBB()` and
`computePerspectiveContextBB()` (`engine/human_limit/nnue_features.h/.cpp`) operating directly on
`chess::bitboard::Position` (two new small public accessors added: `occupiedBitboard()`,
`allOccupiedBitboard()`, `kingSquare()`) instead of a mailbox `BoardArray`. Verified bit-for-bit
identical to the original mailbox implementations across 1,343 positions (20 random self-play walks
+ 9 hand-crafted tactical/pin/double-check/endgame/promotion positions, each walked forward several
plies) via a new permanent test, `tests/test_threat_facts_cross_validate.cpp`, wired into `make
test`. Only after that passed was `Network::evaluateFromAccumulatorsWithThreats()` retargeted to take
the `Position` directly and use both fast paths, eliminating the `toBoardArray()` call from this
function entirely. The original mailbox-based `computeThreatFacts()`/`computePerspectiveContext()`
remain in place as the verified ground truth (the array-based/testing-oracle pattern used
throughout this project).

**Measured result**: full `evaluateFromAccumulatorsWithThreats()` (the actual search hot path) went
from **6.72-6.88us/call to 3.59us/call — roughly 1.9x** faster per node, isolated in
`profile_breakdown`. Full test suite (11 binaries including the new cross-validation test) passes
unchanged. Real-search depth-at-time (30s budget, same two reference positions used throughout this
project):

| position | depth 15 before -> after | depth 16 before -> after |
|---|---|---|
| easy (quiet) | 17.1s -> 10.0s | 27.2s -> **15.1s** (now reaches depth 17 in 24.8s; previously capped at 16) |
| hard (tactical) | 27.4s -> **20.5s** | not reached either way at 30s (big 12->13 branching spike persists — a separate, known search-instability issue, not eval speed) |

A separate, unrelated eval-hot-path cleanup made just before this (removing three per-node heap
allocations in `network.cpp`, replacing them with fixed-size stack arrays sized to a new
`kMaxHidden` compile-time cap) was verified correct but showed **no measurable speed change** in
isolation — that earlier "2x regression" read was a stale apples-to-oranges comparison against a
pre-threat-features baseline, corrected once measured properly. The real, verified win this section
is the bitboard-native threat/perspective computation above.

Also fixed in passing: `Makefile`'s `test_accumulator`/`test_nnue_features` targets were missing
`$(BITBOARD_SRC)` on the link line — a pre-existing gap (both files reference
`chess::bitboard::Position` symbols regardless of whether a given test exercises them), surfaced
when `nnue_features.cpp` gained a hard dependency on `Position` for the new fast path.
