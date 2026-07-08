#include "magic.h"

#include <array>
#include <random>
#include <vector>

namespace chess::bitboard {

namespace {

struct MagicEntry {
    Bitboard mask = 0;
    Bitboard magic = 0;
    int shift = 0;
    std::vector<Bitboard> table;
};

std::array<MagicEntry, 64> bishopMagics;
std::array<MagicEntry, 64> rookMagics;

constexpr std::array<std::pair<int, int>, 4> kBishopDirs = {{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}};
constexpr std::array<std::pair<int, int>, 4> kRookDirs = {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

Bitboard slidingAttacksSlow(int sq, Bitboard occupied, const std::array<std::pair<int, int>, 4>& dirs) {
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

MagicEntry findMagic(int sq, const std::array<std::pair<int, int>, 4>& dirs, std::mt19937_64& rng) {
    Bitboard mask = slidingAttacksSlow(sq, 0, dirs);
    int bits = popcount(mask);
    int tableSize = 1 << bits;
    int shift = 64 - bits;

    std::vector<Bitboard> occupancies(tableSize);
    std::vector<Bitboard> attacks(tableSize);
    Bitboard subset = 0;
    int count = 0;
    do {
        occupancies[count] = subset;
        attacks[count] = slidingAttacksSlow(sq, subset, dirs);
        count++;
        subset = (subset - mask) & mask;
    } while (subset != 0);

    while (true) {
        Bitboard magic = rng() & rng() & rng();
        std::vector<Bitboard> table(tableSize, ~0ULL);
        bool ok = true;
        for (int i = 0; i < count && ok; i++) {
            uint64_t index = (occupancies[i] * magic) >> shift;
            if (table[index] == ~0ULL) {
                table[index] = attacks[i];
            } else if (table[index] != attacks[i]) {
                ok = false;
            }
        }
        if (ok) {
            MagicEntry entry;
            entry.mask = mask;
            entry.magic = magic;
            entry.shift = shift;
            entry.table = std::move(table);
            return entry;
        }
    }
}

}

void initMagics() {
    std::mt19937_64 rng(0xC0FFEE123456789ULL);
    for (int sq = 0; sq < 64; sq++) {
        bishopMagics[sq] = findMagic(sq, kBishopDirs, rng);
        rookMagics[sq] = findMagic(sq, kRookDirs, rng);
    }
}

Bitboard bishopAttacks(int sq, Bitboard occupied) {
    const auto& e = bishopMagics[sq];
    uint64_t index = ((occupied & e.mask) * e.magic) >> e.shift;
    return e.table[index];
}

Bitboard rookAttacks(int sq, Bitboard occupied) {
    const auto& e = rookMagics[sq];
    uint64_t index = ((occupied & e.mask) * e.magic) >> e.shift;
    return e.table[index];
}

}
