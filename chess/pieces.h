#pragma once

#include <array>
#include <optional>
#include <utility>
#include <vector>

namespace chess {

using BoardArray = std::array<std::array<char, 8>, 8>;

enum class Color { White, Black };

struct Pos {
    int r;
    int c;

    bool operator==(const Pos& other) const { return r == other.r && c == other.c; }
};

struct Move {
    Pos from;
    Pos to;
    char promotion = ' ';

    bool operator==(const Move& other) const {
        return from == other.from && to == other.to && promotion == other.promotion;
    }
};

using CastlingRights = std::array<bool, 4>;

extern const std::array<char, 8> LATERAL;

bool onBoard(int r, int c);

bool isAttacked(const BoardArray& board, int r, int c, Color color);

bool isCheck(const BoardArray& board, Color color);

std::vector<Move> genPseudoMoves(const BoardArray& board, Color color,
                                  const CastlingRights& castlingRights,
                                  const std::optional<Pos>& enPassantTarget);

// Precomputed check/pin info for fast legality filtering, as an alternative to make/unmake
// per candidate move. Deliberately does not cover king moves, castling, or en passant capture
// legality (those special cases stay on the make/unmake path — see isLegalFast callers).
struct LegalMoveContext {
    Pos king{-1, -1};
    int checkerCount = 0;
    std::vector<Pos> resolutionSquares;
    std::array<std::array<bool, 8>, 8> pinned{};
    std::array<std::array<std::pair<int, int>, 8>, 8> pinDir{};
};

LegalMoveContext computeLegalMoveContext(const BoardArray& board, Color color);
// Same, but skips the internal king scan when the caller already knows the king's square.
LegalMoveContext computeLegalMoveContext(const BoardArray& board, Color color, Pos king);

// Only valid for non-king, non-en-passant-capture moves; see LegalMoveContext.
bool isLegalFast(const LegalMoveContext& ctx, const Move& m);

}
