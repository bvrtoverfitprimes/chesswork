# Research notes

Findings from web-research agents dispatched during the human_limit NNUE/search overhaul.
Kept verbatim (with sources) so nothing gets lost across a long session — treat as reference
material for what to implement next, not as already-implemented.

## 1. Stockfish NNUE architecture & training recipe (researched before the H=384/bucket rewrite)

**Architecture specifics:**
- Feature set: `HalfKAv2_hm` for SFNNv7-v11; most recent generations (SFNNv12-13) moved to a
  larger "Full_Threats+" feature set including threat info, SFNNv13 at 82,672 features
  ([DeepWiki NNUE arch reference](https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference)).
  HalfKAv2 differs from HalfKP (what we use) by also encoding the king's own square as a
  feature slot, plus horizontal-mirror-only symmetry (`_hm`) instead of full 8-way
  ([commit e8d64af](https://github.com/official-stockfish/Stockfish/commit/e8d64af1230fdac65bb0da246df3e7abe82e0838)).
- Layer widths evolved: `256x2->32->32->1` -> `512x2->16->32->1` -> current-generation
  accumulators on the order of **1024 per perspective**, small L2 (8-16), L3 (32).
- Activation: **Clipped ReLU** `y = min(max(x,0),1)` (or `[0,127]` quantized), used because
  "aggressive quantization requires reducing the dynamic range of hidden layer inputs"
  (official nnue.md). **Squared Clipped ReLU (SCReLU)** on the first-layer output is used in
  some modern nets — lower-confidence, not directly quoted from a primary doc paragraph.
  *We deliberately did NOT adopt Clipped ReLU*: its rationale is quantization-driven, and our
  C++ inference is plain unquantized `double` — copying the clamp range without the matching
  quantization step could bottleneck an unquantized model for no benefit. Kept plain ReLU.
- **Output buckets (confirmed, adopted)**: 8 output subnetworks, bucket index =
  `(popcount(all pieces on board) - 1) / 4`, each with its own small head, only one evaluated
  per position (marginal speed cost)
  ([commit e8d64af](https://github.com/official-stockfish/Stockfish/commit/e8d64af1230fdac65bb0da246df3e7abe82e0838)).
  This is exactly what `engine/human_limit/nnue_features.cpp:computeOutputBucket` and the
  bucketed head in `training/model/net.py` implement now.

**Training loss (confirmed from official `docs/nnue.md`, adopted):**
- CP -> WDL: `wdl = sigmoid(cp / scaling_factor)`, example `scaling_factor ~= 410` (our `/400`
  is close and reasonable).
- Full formula blends `lambda * wdl_eval_target + (1-lambda) * game_result` — we don't have
  game-result labels (no self-play games tied to our Lichess/Stockfish cp-only data), so we
  effectively run with `lambda=1` (pure eval-based target). **Not yet exploited**: if we ever
  add self-play games with known outcomes, this blend is the documented next step.
- Loss: squared error in WDL space, but **exponent 2.6 instead of 2** is documented as
  what Stockfish nets actually train well with. Adopted in `training/model/train.py::wdl_loss`.

**Data volume (confirmed):** current Stockfish training sets are on the order of **10+
billion positions** (one figure: 16 billion), shallow-depth (8-12) self-play plus Lc0-sourced
data. **No public figures found** for network quality at our scale (hundreds of thousands to
low millions of positions) — flagged explicitly as an unfilled gap, not guessed.

**Pitfalls (moderate confidence, from nnue-pytorch wiki/community):** quantization-aware
clamping necessary for float->int8/16 transfer (not applicable to us, no quantization);
LR warmup/step decay is standard (we added StepLR decay); position diversity/deduplication is
a real quality lever (we mix Lichess bulk data with a targeted curriculum, see below);
**validation-loss divergence is the standard overfitting signal** — this is exactly what we
hit and fixed (see WORKLOG section on the M2 training run: first run overfit past epoch 6,
caught via val_mae, fixed by adding best-checkpoint tracking + early stopping to `train.py`).

**Expected Elo at our scale:** no primary-source numbers found; explicitly unquantified.

Sources: [nnue.md](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md),
[NNUE Stockfish Docs](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html),
[commit e8d64af](https://github.com/official-stockfish/Stockfish/commit/e8d64af1230fdac65bb0da246df3e7abe82e0838),
[DeepWiki NNUE arch reference](https://deepwiki.com/official-stockfish/nnue-pytorch/9-nnue-architecture-reference),
[Training datasets wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets).

## 2. Search pruning/move-ordering techniques (researched to close the effective-branching-factor gap)

Context that motivated this research: measured Stockfish reaching depth 18-20 in 1s
(effective branching factor ~1.94) on this machine, vs. our depth 5-7 (branching factor
5.8-8.7), despite Stockfish's raw nodes/sec being only 2.5-10x higher — i.e. the gap is
dominated by search efficiency (pruning/ordering), not eval speed.

**1. Futility Pruning (FP) & Reverse Futility Pruning (RFP / static null move)**

RFP (non-PV nodes near the tips, typically depth <= 8-9): if `staticEval - margin(depth) >=
beta`, return early. Stockfish's live formula (`search.cpp`):
```
futilityMult = min(40 + depth*4, 80) - 20*(!ttHit)
margin = futilityMult*depth - (2934*improving + 343*opponentWorsening)*futilityMult/1024 + |correctionValue|/182069
if (eval - margin >= beta) return (716*beta + 308*eval)/1024
```
Simpler classical formula widely cited on chessprogramming.org: `margin ~= 150 * depth`.
We don't have "improving"/correction-history state, so start with the simple classical
margin and only adopt Stockfish's exact formula if the simple version proves insufficient.

Forward FP (main search, non-PV, not in check): if `staticEval + margin(depth) <= alpha` for
a *quiet* move, skip it. Classic per-ply margins: depth1~=100, depth2~=300, depth3~=500cp.
Never applies to checks, promotions, or captures.

**2. Late Move Pruning (LMP)** — distinct from LMR: once `moveCount` exceeds a
depth-dependent threshold for quiet, non-check, non-killer moves, skip the rest entirely.
Stockfish live condition: `moveCount >= (3 + depth*depth) / (2 - improving)`. Older
widely-copied table (two rows, improving/non-improving):
```
counts[improving][depth] = {0,3,4,6,10,14,19,25,31} / {0,5,7,11,17,26,36,48,63}
```

**3. Static Exchange Evaluation (SEE)** — board-representation-agnostic swap-list algorithm:
repeatedly find the least-valuable attacker of a target square (alternating sides), push
captured value onto a gain list, then negamax the list backward:
`gain[k-1] = -max(-gain[k-1], gain[k])`; final `gain[0]` is the SEE score. Two uses:
- Quiescence: prune captures with `SEE < 0` before searching (paired with delta pruning).
- Main search ordering: order good captures (SEE>=0) by SEE/MVV-LVA ahead of quiet moves;
  push losing captures (SEE<0) to the back, after killers/history.
Correctness trap specific to our array-based (non-bitboard) engine: SEE must re-derive
x-ray/pin attackers correctly after each simulated removal — a pinned "recapture" piece may
not legally be able to recapture, and array-based SEE has to re-scan sliding lines manually
each iteration (no free bitboard "attackers to square" recompute).

**4. ProbCut** — shallow null-window verification search predicts a full-depth fail
high/low, used to prune main-search nodes at moderate depth. Stockfish live params:
```
probCutBeta = beta + 214 - 59*improving
probCutDepth = depth - 4 - improving
```
Noted: in chess, ProbCut competes with null-move pruning for the same cutoff type and is
less dominant than in Othello — smaller gains than null-move alone. Lower priority than
RFP/LMP/SEE for us.

**5. Singular Extensions** — verify the TT move is uniquely good via a reduced-depth,
narrow-window research of *other* moves below the TT score; if none beats it, extend the TT
move. Stockfish trigger: `depth>=6`, TT has a lower bound, `|ttValue|` not mate-range,
`ttData.depth >= depth-3`; `singularBeta = ttValue - (60 + 70*(pv&&!PvNode)) * depth / 59`,
verification search at roughly `(depth-1)/2`. Needs an `excludedMove` mechanism to avoid
infinite/circular re-verification, and extension-count caps. Higher implementation
complexity/risk than RFP/LMP/SEE — candidate for a later pass, not this one.

**6. Other contributors, lower priority for this pass**
- **Internal Iterative Reductions (IIR)**: no TT move + depth>5 -> reduce that node's depth
  by 1 rather than doing a full internal search. Cheap, Stockfish-adopted 2021 (from Rebel).
- **Correction History**: tracks systematic static-eval bias (indexed by pawn structure/
  material), adjusts staticEval before all pruning decisions. Recent (Stockfish 17+) major
  contributor — meaningful but adds real state/complexity; candidate for a later pass.
- **Multi-Cut**: reduced-depth search of first 3-6 moves at expected cut-nodes; if >=2 fail
  high, prune the whole node. Less common in modern top engines than RFP/LMP.
- **Razoring**: at low depth (~1-3), if staticEval far below alpha, drop into qsearch
  directly. Stockfish: `eval < alpha - 465 - 300*depth*depth`.

**7. Realistic expectations**: well-documented (chessprogramming.org "Pruning" overview,
multiple engine changelogs) that RFP + LMP + real SEE ordering + futility pruning are the
dominant EBF-reducing techniques, largely *independent of bitboards/threading* — tree-shape
optimizations, not raw-speed optimizations. Several historical non-bitboard engines (early
Fruit, Toga, some array-based Java/C# engines) reached depth 12-15 in similar time using
exactly this technique stack. Matches our diagnosis: bitboards mainly buy raw NPS (the
2.5-10x we measured), not effective branching factor.

**Correctness traps flagged (critical, must respect in implementation)**:
- Never apply RFP/futility/LMP in-check, on a move that gives check, or on promotions —
  these can hide mate threats and cause severe blunders.
- Static eval must be stable/correct at the pruning point — verify accumulator/bucket state
  is exactly right before trusting it for a pruning decision (matters a lot given our
  incremental-accumulator design).
- LMP/futility must never prune away the *last* remaining legal move at a node (mate/
  stalemate correctness) — always ensure at least one move is fully searched before any
  pruning-driven early return, or fall back to full search if nothing was searched.
- Singular extension search must exclude the singular move itself to avoid infinite/circular
  re-verification.

**8. Novel NN-signal-as-pruning-input ideas**: no credible primary source found describing
alpha-beta pruning conditioned on a network's own value confidence/uncertainty or eval
volatility (as opposed to using the NN purely as leaf evaluation). Tangential MCTS-family
ideas exist (KataGo policy-target pruning, entropy-regularized MCTS) but those are
PUCT/MCTS techniques, not alpha-beta. Flagged explicitly as *not confirmed to exist* as an
established technique — this is genuinely open territory if we want to try something novel,
not a case of us missing known prior art.

Sources: chessprogramming.org pages for Reverse_Futility_Pruning, Futility_Pruning,
Late_Move_Reductions, Static_Exchange_Evaluation, SEE_-_The_Swap_Algorithm, ProbCut,
Singular_Extensions, Internal_Iterative_Reductions, Multi-Cut, Pruning (overview);
[live Stockfish search.cpp](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp);
[talkchess SEE/qsearch discussion](https://talkchess.com/viewtopic.php?t=41217);
[Stockfish 18 blog post, Correction History mention](https://stockfishchess.org/blog/2026/stockfish-18/).
Confidence: high for Stockfish-sourced live formulas, medium-high for wiki-only claims (direct
chessprogramming.org fetches timed out at the network level this session; corroborated via
search-engine snippets of the same pages).

## 3. Plan for this implementation pass

Priority order (highest value / lowest risk first), all confined to `engine/human_limit/`:
1. Real SEE (swap-algorithm) — improves move ordering immediately, and is a prerequisite for
   quiescence SEE-pruning.
2. Quiescence SEE-pruning — skip searching captures with negative SEE.
3. Reverse futility pruning (simple `150*depth` margin to start).
4. Forward futility pruning (per-ply margin table) for quiet moves near the frontier.
5. Late move pruning (move-count table above).
6. One novel idea tailored to our bucketed-NNUE architecture (see WORKLOG for what was
   actually tried) — flagged explicitly as experimental since no prior art confirms it works.

ProbCut, singular extensions, correction history, multi-cut: deferred as a follow-up pass —
higher complexity/risk-to-reward ratio for a first implementation, per the research above.

## 4. Deep-dive follow-up (after the first pruning pass landed: SEE, RFP, forward futility,
LMP — measured result: depth 8 @ 1s, effective branching factor ~4.6, vs Stockfish's depth
18-20 @ EBF ~1.94 on the same machine). Researched the deferred items in more depth, live
Stockfish source, checked July 2026.

**Correction history** (highest-priority next item per community/commit-history consensus —
one of the largest single-patch Elo gains in Stockfish's history at introduction): four+
independent int16 tables per side, each hashed and updated the same way
(`bonus = clamp((bestValue - staticEval) * depth / 8, ±LIMIT/4)`), blended with fixed weights
before being added to static eval — a "shrunk consensus vote" from multiple structural
hashes, not just pawn structure:
- Pawn structure: keyed by `pos.pawn_key()`
- Minor-piece placement
- Non-pawn material/placement, tracked *separately* for White and Black pieces
- Continuation-correction: indexed by `(piece, to-square)` from 2 and 4 plies back — same
  indexing shape as a normal continuation-history table, but correcting eval error instead of
  scoring move ordering.
Combine: `correction = (13345*pawnCorr + 9280*minorCorr + 11840*(whiteNonPawnCorr +
blackNonPawnCorr) + continuationCorr) / 131072`, added to `staticEval` **before** RFP/
futility/ProbCut margins are computed. Updated after each search node returns (excluded when
in check). Table blend weights may drift slightly version to version — structure is stable.
Directly cheapens our *existing* RFP/futility decisions (sharper margins from a
less-systematically-biased static eval) rather than requiring new search machinery — matches
why it's ranked the best next win given we already have RFP/futility/LMR/null-move.

**Singular extensions** (second priority — increases search *effort* per depth for
score-critical lines rather than raw EBF, i.e. strength-per-node not nodes-per-depth):
```
singularBeta = ttValue - (60 + 70*(ttPv && !PvNode)) * depth / 59
singularDepth = newDepth / 2   // verification search excludes the TT move
if (verificationValue < singularBeta):
    doubleMargin = -3 + 201*PvNode - 157*!ttCapture - (correction-history term)
    tripleMargin = 72 + 306*PvNode - 188*!ttCapture + 84*(ttPv term)
    extension = 1 + (value < singularBeta - doubleMargin) + (value < singularBeta - tripleMargin)  // 1-3 plies
elif singularBeta >= beta: return singularBeta   // multi-cut lives here now, not separate
elif ttValue >= beta: extension = -2   // negative extension
elif cutNode: extension = -1
```
Notably: **multi-cut is no longer a separate technique in current Stockfish — it's subsumed
into the singular-extension verification search** (the `singularBeta >= beta` branch prunes
the whole node). Needs an `excludedMove` mechanism on the verification search and
double-extension-count tracking to bound tree blowup (nodes that got 2+ ply extended suppress
further LMR-based extension downstream).

**ProbCut**: `probCutBeta = beta + 214 - 59*improving` (constants seen to vary ~±20 across
recent commits — structure stable, treat exact constants as tunable). `probCutDepth` was
recently simplified to `clamp(depth - 5 - (staticEval - beta)/315, 0, depth)` — factors in
*how far* static eval is from beta, not just a flat `depth - 4 - improving` like the version
in section 2 above. Trigger: `depth >= 3`, beta not already a decisive/mate score, and not
already known-bad from a TT hit below probCutBeta. Mechanism: qsearch probe ordered by
capture history at `probCutBeta - staticEval`; if it clears, a full search at `probCutDepth`
confirms.

**Internal Iterative Reductions (IIR)**: `if (!followingPV && !allNode && depth >= 6 &&
!ttMove) depth--;` — flat 1-ply reduction, PV/Cut nodes only, additionally gated by
`priorReduction <= 3` so it doesn't stack on already-reduced branches. Simple, nearly free,
lowest implementation cost of the four.

**The `improving` flag** (needed by RFP/ProbCut formulas above, not currently tracked):
`improving = staticEval(ply) > staticEval(ply-2)` (same side to move, hence 2 plies back, not
1), plus a related `opponentWorsening = staticEval(ply) > -staticEval(ply-1)`. When ply-2 data
is unreliable (e.g. was a check-evasion node), Stockfish chains back further (ply-4) rather
than special-casing a boolean fallback. Implementable by threading `staticEval` through a
per-ply search-stack array (we don't currently have one — `negamax` only takes `ply` as an
int, not a stack struct) instead of the current ad hoc `getStaticEval()` lambda scoped to one
call.

**NNUE `Full_Threats+` feature set** (SFNNv12-13, ~82,672 total inputs): lower confidence —
public docs describe its *existence* (a `threats.yaml` training config) but not the precise
threat-relation encoding (attacked-by/defended-by/pinned, and how they're bucketed into the
HalfKA-style index) in any fetchable primary source. Practically: adoptable in principle since
we already compute attacks for SEE, but reworking the feature-index function to add a second
"threat-plane" (roughly doubling active feature count and embedding table size) is a real
architecture project, not a quick win. **Deprioritized relative to the search items above** —
our current bottleneck is search depth/EBF (4.6 vs 1.94), not eval accuracy, so this doesn't
address the thing actually limiting us right now.

**Confirmed, adoptable without quantization**: current Stockfish hidden layers use **Squared
Clipped ReLU** (`clamp(x, 0, 127)²`) rather than plain Clipped ReLU in the deeper layers — a
real, cheap, drop-in activation swap on plain float32 (no quantization required), worth
testing separately from the search-side work above.

**Priority order for the next implementation pass**: correction history first (cheapens
existing pruning, historically one of the largest single-patch Elo gains), singular
extensions second (real but different kind of win — node-effort not EBF), IIR third (cheap,
small), ProbCut fourth (narrower scope, mainly non-PV nodes). `Full_Threats+` and the
activation-function swap are separate, lower-priority tracks.

Sources: live Stockfish `search.cpp` (github.com/official-stockfish/Stockfish), commit
b4d995d (correction history introduction), commit 77d46ff and PR #5697 (ProbCut depth-term
simplification), commit cef5510 (IIR), chessprogramming.org (Internal_Iterative_Reductions),
nnue-pytorch docs/nnue.md, Stockfish `clipped_relu.h`/`sqr_clipped_relu.h`, TalkChess thread
on correction history's introduction (viewtopic.php?t=35419).
