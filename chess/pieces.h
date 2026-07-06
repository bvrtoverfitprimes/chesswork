#pragma once

#include <array>
#include <optional>
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
};

using CastlingRights = std::array<bool, 4>;

extern const std::array<char, 8> LATERAL;

bool onBoard(int r, int c);

bool isAttacked(const BoardArray& board, int r, int c, Color color);

bool isCheck(const BoardArray& board, Color color);

std::vector<Move> genPseudoMoves(const BoardArray& board, Color color,
                                  const CastlingRights& castlingRights,
                                  const std::optional<Pos>& enPassantTarget);

} // namespace chess
