#include "bitboard.h"

namespace chess::bitboard {

std::array<Bitboard, 64> knightAttacks{};
std::array<Bitboard, 64> kingAttacks{};
std::array<Bitboard, 64> whitePawnAttacks{};
std::array<Bitboard, 64> blackPawnAttacks{};
std::array<std::array<Bitboard, 64>, 64> squaresBetween{};

namespace {

constexpr std::array<std::pair<int, int>, 8> kKnightOffsets = {{
    {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
}};

constexpr std::array<std::pair<int, int>, 8> kKingOffsets = {{
    {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}
}};

}

void initAttackTables() {
    for (int sq = 0; sq < 64; sq++) {
        int f = fileOf(sq), r = rankOf(sq);

        Bitboard n = 0;
        for (auto [df, dr] : kKnightOffsets) {
            if (onBoardFR(f + df, r + dr)) n |= (1ULL << squareOf(f + df, r + dr));
        }
        knightAttacks[sq] = n;

        Bitboard k = 0;
        for (auto [df, dr] : kKingOffsets) {
            if (onBoardFR(f + df, r + dr)) k |= (1ULL << squareOf(f + df, r + dr));
        }
        kingAttacks[sq] = k;

        Bitboard wp = 0;
        if (onBoardFR(f - 1, r + 1)) wp |= (1ULL << squareOf(f - 1, r + 1));
        if (onBoardFR(f + 1, r + 1)) wp |= (1ULL << squareOf(f + 1, r + 1));
        whitePawnAttacks[sq] = wp;

        Bitboard bp = 0;
        if (onBoardFR(f - 1, r - 1)) bp |= (1ULL << squareOf(f - 1, r - 1));
        if (onBoardFR(f + 1, r - 1)) bp |= (1ULL << squareOf(f + 1, r - 1));
        blackPawnAttacks[sq] = bp;
    }

    constexpr std::array<std::pair<int, int>, 8> kAllDirs = {{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    }};
    for (int a = 0; a < 64; a++) {
        for (auto [df, dr] : kAllDirs) {
            int f = fileOf(a) + df, r = rankOf(a) + dr;
            Bitboard between = 0;
            while (onBoardFR(f, r)) {
                int b = squareOf(f, r);
                squaresBetween[a][b] = between;
                between |= (1ULL << b);
                f += df;
                r += dr;
            }
        }
    }
}

}
