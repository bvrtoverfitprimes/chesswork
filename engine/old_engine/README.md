# old_engine

The classical, handcrafted state-of-the-art reproduction (namespace
`old_engine`).

Evaluation: tapered piece-square tables (the well-known "PeSTO"-style
midgame/endgame tables) plus material, blended by a computed game
phase so the engine values pieces differently in the opening/middlegame
versus the endgame (e.g. a centralized king is penalized early, but
rewarded once the position simplifies). Every number in these tables
is a fixed, hand-chosen constant from established chess programming
literature — nothing here is learned or trained.

Search: negamax with alpha-beta pruning, iterative deepening under a
time budget, a transposition table, quiescence search (with proper
check-evasion handling), null-move pruning, late move reductions,
aspiration windows, and layered move ordering (transposition-table
move, then MVV-LVA captures, then killer moves, then history
heuristic).

This is the "handcrafted eval + classical search" recipe — the same
shape every top engine used before neural evaluation (NNUE) took over,
and still the search skeleton underneath NNUE-era engines today. Kept
as the last fully deterministic, fully human-understandable baseline
before the project moved to a learned evaluation.
