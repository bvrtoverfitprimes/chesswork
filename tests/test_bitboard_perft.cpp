#include <cstdint>
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

uint64_t perft(chess::bitboard::Position& pos, int depth) {
    if (depth == 0) return 1;
    uint64_t count = 0;
    auto moves = pos.getValidMoves();
    for (const auto& m : moves) {
        char promo = (m.promotion == ' ') ? 'q' : m.promotion;
        auto undo = pos.makeMove(m.from, m.to, promo);
        count += perft(pos, depth - 1);
        pos.unmakeMove(undo);
    }
    return count;
}

}

int main() {
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    {
        chess::bitboard::Position pos;
        check(perft(pos, 1) == 20, "perft startpos depth 1 = 20");
        check(perft(pos, 2) == 400, "perft startpos depth 2 = 400");
        check(perft(pos, 3) == 8902, "perft startpos depth 3 = 8902");
        check(perft(pos, 4) == 197281, "perft startpos depth 4 = 197281");
        check(perft(pos, 5) == 4865609, "perft startpos depth 5 = 4865609");
    }

    {
        chess::bitboard::Position pos("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        check(perft(pos, 1) == 48, "perft kiwipete depth 1 = 48");
        check(perft(pos, 2) == 2039, "perft kiwipete depth 2 = 2039");
        check(perft(pos, 3) == 97862, "perft kiwipete depth 3 = 97862");
    }

    {
        // Position 3 from the standard perft test suite (endgame-heavy, exercises en passant).
        chess::bitboard::Position pos("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
        check(perft(pos, 1) == 14, "perft position3 depth 1 = 14");
        check(perft(pos, 2) == 191, "perft position3 depth 2 = 191");
        check(perft(pos, 3) == 2812, "perft position3 depth 3 = 2812");
        check(perft(pos, 4) == 43238, "perft position3 depth 4 = 43238");
    }

    {
        // Position 4 from the standard perft test suite (castling + promotion heavy).
        chess::bitboard::Position pos("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1");
        check(perft(pos, 1) == 6, "perft position5 depth 1 = 6");
        check(perft(pos, 2) == 264, "perft position5 depth 2 = 264");
        check(perft(pos, 3) == 9467, "perft position5 depth 3 = 9467");
    }

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
