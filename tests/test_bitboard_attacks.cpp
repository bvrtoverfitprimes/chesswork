#include <array>
#include <iostream>
#include <random>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"

namespace {

using namespace chess::bitboard;

int failures = 0;
int checks = 0;

void check(bool cond, const std::string& label) {
    if (!cond) {
        std::cout << "FAIL: " << label << "\n";
        failures++;
    } else {
        checks++;
    }
}

Bitboard bruteForceSliding(int sq, Bitboard occupied, const std::array<std::pair<int, int>, 4>& dirs) {
    Bitboard attacks = 0;
    int f0 = fileOf(sq), r0 = rankOf(sq);
    for (auto [df, dr] : dirs) {
        int f = f0 + df, r = r0 + dr;
        while (onBoardFR(f, r)) {
            int s = squareOf(f, r);
            attacks |= (1ULL << s);
            if (occupied & (1ULL << s)) break;
            f += df;
            r += dr;
        }
    }
    return attacks;
}

Bitboard bruteForceKnight(int sq) {
    static constexpr std::array<std::pair<int, int>, 8> offsets = {{
        {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
    }};
    Bitboard b = 0;
    int f = fileOf(sq), r = rankOf(sq);
    for (auto [df, dr] : offsets) {
        if (onBoardFR(f + df, r + dr)) b |= (1ULL << squareOf(f + df, r + dr));
    }
    return b;
}

Bitboard bruteForceKing(int sq) {
    Bitboard b = 0;
    int f = fileOf(sq), r = rankOf(sq);
    for (int df = -1; df <= 1; df++) {
        for (int dr = -1; dr <= 1; dr++) {
            if (df == 0 && dr == 0) continue;
            if (onBoardFR(f + df, r + dr)) b |= (1ULL << squareOf(f + df, r + dr));
        }
    }
    return b;
}

}

int main() {
    initAttackTables();
    initMagics();

    constexpr std::array<std::pair<int, int>, 4> kBishopDirs = {{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};
    constexpr std::array<std::pair<int, int>, 4> kRookDirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

    for (int sq = 0; sq < 64; sq++) {
        check(knightAttacks[sq] == bruteForceKnight(sq), "knight attacks sq=" + std::to_string(sq));
        check(kingAttacks[sq] == bruteForceKing(sq), "king attacks sq=" + std::to_string(sq));
    }

    std::mt19937_64 rng(1234);
    for (int trial = 0; trial < 5000; trial++) {
        int sq = rng() % 64;
        Bitboard occ = rng() & rng();

        Bitboard expectedBishop = bruteForceSliding(sq, occ, kBishopDirs);
        Bitboard expectedRook = bruteForceSliding(sq, occ, kRookDirs);
        check(bishopAttacks(sq, occ) == expectedBishop, "bishop attacks sq=" + std::to_string(sq) + " trial=" + std::to_string(trial));
        check(rookAttacks(sq, occ) == expectedRook, "rook attacks sq=" + std::to_string(sq) + " trial=" + std::to_string(trial));
    }

    std::cout << checks << " checks passed, " << failures << " failures\n";
    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
