# Work Log

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
