#pragma once

#include <array>
#include <cstdint>

// Square numbering here is the standard little-endian rank-file (LERF) convention used by
// virtually all published bitboard/magic-number material: square 0 = a1, square 63 = h8,
// sq = rank*8 + file (rank 0 = rank 1, file 0 = file a). This is deliberately different from
// chess::Pos's (r,c) convention (r=0 = rank 8) used by the existing array engine — the two
// are only ever bridged explicitly, via toSquare/fromSquare-style helpers, at the boundary
// (Position::toBoardArray and friends), never mixed silently.

namespace chess::bitboard {

using Bitboard = uint64_t;

constexpr Bitboard kFileA = 0x0101010101010101ULL;
constexpr Bitboard kFileH = 0x8080808080808080ULL;
constexpr Bitboard kRank1 = 0x00000000000000FFULL;
constexpr Bitboard kRank8 = 0xFF00000000000000ULL;

constexpr int fileOf(int sq) { return sq & 7; }
constexpr int rankOf(int sq) { return sq >> 3; }
constexpr int squareOf(int file, int rank) { return rank * 8 + file; }
constexpr bool onBoardFR(int file, int rank) { return file >= 0 && file < 8 && rank >= 0 && rank < 8; }

inline int popcount(Bitboard b) { return __builtin_popcountll(b); }
inline int bitscanForward(Bitboard b) { return __builtin_ctzll(b); }
inline int popLsb(Bitboard& b) {
    int i = bitscanForward(b);
    b &= b - 1;
    return i;
}

extern std::array<Bitboard, 64> knightAttacks;
extern std::array<Bitboard, 64> kingAttacks;
extern std::array<Bitboard, 64> whitePawnAttacks;
extern std::array<Bitboard, 64> blackPawnAttacks;
// squaresBetween[a][b]: squares strictly between a and b if they share a rank/file/diagonal
// (exclusive of both endpoints), else 0. Used for O(1) pin detection and check-block squares.
extern std::array<std::array<Bitboard, 64>, 64> squaresBetween;

void initAttackTables();

}
