#pragma once

#include "bitboard.h"

namespace chess::bitboard {

// Generates magic numbers by construction (random trial + collision check against a
// brute-force ground truth), rather than relying on hardcoded published constants — this is
// self-verifying: initMagics() cannot complete with an incorrect table, since any magic
// candidate that produces a hash collision between two different attack sets is rejected and
// retried. Runs in well under a second (fixed seed, deterministic).
void initMagics();

Bitboard bishopAttacks(int sq, Bitboard occupied);
Bitboard rookAttacks(int sq, Bitboard occupied);
inline Bitboard queenAttacks(int sq, Bitboard occupied) {
    return bishopAttacks(sq, occupied) | rookAttacks(sq, occupied);
}

}
