#include <iostream>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"

namespace {

int failures = 0;

void check(bool cond, const std::string& label) {
    if (!cond) {
        std::cout << "FAIL: " << label << "\n";
        failures++;
    } else {
        std::cout << "PASS: " << label << "\n";
    }
}

int seeOf(const std::string& fen, const std::string& fromSq, const std::string& toSq) {
    chess::bitboard::Position pos(fen);
    int from = chess::bitboard::Position::parseSquareUci(fromSq);
    int to = chess::bitboard::Position::parseSquareUci(toSq);
    return pos.see(from, to);
}

}

int main() {
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    check(seeOf("4k3/8/8/8/8/8/1p6/1R2K3 w - - 0 1", "b1", "b2") == 100,
          "rook takes undefended pawn (same file), SEE=+100");

    check(seeOf("4k3/8/8/4p3/3n4/2P5/8/4K3 w - - 0 1", "c3", "d4") == 320 - 100,
          "pawn takes knight defended by pawn, SEE=+220");

    check(seeOf("4k3/8/8/8/r7/8/8/R3K3 w - - 0 1", "a1", "a4") == 500,
          "rook takes completely undefended rook (same file), SEE=+500");

    check(seeOf("3r1k2/8/8/8/3r4/8/8/3RK3 w - - 0 1", "d1", "d4") == 0,
          "rook takes rook defended by another rook behind it, even trade SEE=0");

    check(seeOf("4k3/3q4/8/8/R2n4/8/8/3K4 w - - 0 1", "a4", "d4") == 320 - 500,
          "rook takes knight defended by queen behind it (same file) — losing capture, SEE=-180");

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
