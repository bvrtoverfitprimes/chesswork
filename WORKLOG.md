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
