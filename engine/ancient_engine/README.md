# ancient_engine

The original MVP engine (namespace `ancient_engine`).

Evaluation: material count (pawn=100, knight=320, bishop=330, rook=500,
queen=900) plus a flat +10 bonus for any non-king piece occupying one
of the four center squares. No piece-square tables, no game-phase
awareness, no positional knowledge beyond that single center bonus.

Search: plain negamax with alpha-beta pruning at a fixed depth (no
iterative deepening, no transposition table, no quiescence search, no
move ordering beyond a simple capture-first sort). Every node is
searched to the same fixed depth regardless of how tactical or quiet
the position is, so it is prone to the horizon effect (misjudging a
position that is mid-capture-sequence right at the search cutoff).

Kept only as a historical reference point for how far the project has
come.
