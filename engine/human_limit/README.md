# human_limit

The current engine (namespace `human_limit`). Evaluation is a mix of a
trained neural network and a hand-written classical formula; search is
the same classical alpha-beta skeleton used in `old_engine`.

## Evaluation: a mixed architecture

The final evaluation is a weighted blend:

```
score = 0.3 * classicalEval + 0.7 * networkEval
```

`classicalEval` is the exact same tapered piece-square-table formula
used in `old_engine` (material + midgame/endgame PST tables,
interpolated by game phase) — deterministic, hand-set, always
structurally correct. `networkEval` is a small feedforward network: 20
engineered input features -> hidden layer of 128 (ReLU) -> hidden layer
of 64 (ReLU) -> hidden layer of 32 (ReLU) -> a single linear output.
Roughly 13,000 trainable weights, all learned, not hand-set.

Why mix the two rather than use the network alone: early training runs
produced a network that, even after real training, would still fail to
recognize a completely free hanging rook as good to capture — in both
a bare king+rook endgame and a dense middlegame position with a full
army still on the board (confirmed via direct evaluation, not just
search output, so it was a genuine value-function gap, not a search
bug). The likely cause is that self-play games starting from the
normal starting position rarely organically produce such a stark,
clean material imbalance — a shallow searcher rarely just blunders a
whole piece — so the network had very little training signal for that
exact pattern and its behavior there was largely unconstrained
extrapolation. Blending in the always-reliable classical evaluation
gives the engine a structural floor: even where the learned component
hasn't generalized, the hand-crafted 30% still correctly rewards
obvious material and positional gains. Confirmed this fixes both
failing test positions immediately, even before any additional
retraining.

The 20 network input features are engineered (material counts, the
tapered mg/eg/phase values also used directly as `classicalEval`,
mobility, king safety, pawn structure — doubled/isolated/passed pawns,
bishop pair, rook on open/semi-open files, advanced knights, king file
exposure, central pawn control, and side to move) rather than a raw
sparse per-square encoding. This is a deliberate scope decision: a full
sparse NNUE-style input (768+ binary features) needs an *incrementally
updated* accumulator threaded through every make/unmake in the search
to stay fast — real NNUE engines recompute only the few features that
change per move rather than the whole input every time. Building that
incremental-update integration is real engineering work beyond this
generation's scope. Recomputing this compact feature vector fresh at
every node keeps inference cheap without that integration, at the cost
of using less raw positional information than a full NNUE input would.

## Search

Identical in structure to `old_engine`: negamax with alpha-beta
pruning, a transposition table, iterative deepening under a time
budget, quiescence search (with check-evasion handling), null-move
pruning, late move reductions, aspiration windows, and layered move
ordering (transposition-table move, then MVV-LVA captures, then killer
moves, then history heuristic). The only thing that changed from
`old_engine` is what function is called to score a leaf position.

## Training

Trained by self-play reinforcement learning — the same training
*paradigm* AlphaZero and Leela Chess Zero use (the network's own
component only ever learns from games it plays against itself plus a
small amount of synthetic auxiliary data generated locally, no
external game data, no other engine's evaluations used anywhere),
scaled down to a much smaller network and training budget aimed at a
much more modest strength target. See `train_human_limit.cpp` at the
repo root:

1. Play a batch of self-play games using the current network + the
   search above, with a short per-move time budget (data generation
   needs to be fast, not strong).
2. The first several plies of each game and roughly 8% of moves
   throughout are chosen randomly rather than by search, so the
   generated games explore a variety of positions instead of repeating
   the same deterministic line every time.
3. Every third ply's feature vector is recorded, labeled with the
   game's actual final result (win/loss/draw) once it's known —
   including games that hit the ply cap without checkmate (labeled as
   a draw) rather than discarding them, since discarding them was
   found to throw away most of a generation's data.
4. A batch of synthetic "obvious material imbalance" positions is also
   generated each generation: play a handful of random plies from the
   start position for natural piece variety, then remove 1-3 random
   non-king pieces from one side, and label the position by which side
   now has more material. This directly and repeatedly teaches the
   pattern that self-play alone rarely produces on its own.
5. All of this is appended to a persistent replay buffer (capped at a
   fixed size, oldest evicted first) rather than trained on and
   discarded — training only on each generation's small fresh batch
   was found to be nowhere near enough consistent signal for the
   network to reliably learn something as basic as "material
   difference matters."
6. Run several epochs of plain SGD (hand-written forward and backward
   pass, mean-squared-error loss toward the game result) over the
   whole replay buffer.
7. Save the updated weights and repeat — each generation's self-play
   games are generated by the previous generation's just-updated
   network.

Note: the network's training target is the *unscaled* game result
(-1/0/+1), matching `forward()`'s native output range — an earlier
version trained against a target scaled by 1000 (matching
`evaluate()`'s external centipawn-like output) while training on the
raw unscaled `forward()` value, a mismatch that produced huge
first-step gradients and corrupted the learned weights into responding
backwards to material features. Fixed by keeping training entirely in
the network's native unscaled range; the 1000x scaling only happens in
`evaluate()`, after training.

Weights are stored in `weights.txt` (plain text, one float per line,
not committed to git — see `.gitignore` — since it's a training
artifact, not source code).
