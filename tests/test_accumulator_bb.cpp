#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../engine/human_limit/accumulator.h"
#include "../engine/human_limit/network.h"

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

double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) m = std::max(m, static_cast<double>(std::abs(a[i] - b[i])));
    return m;
}

void runRandomWalk(const human_limit::Network& net, const std::string& fen, int plies, uint32_t seed,
                    const std::string& label) {
    chess::bitboard::Position pos(fen);
    human_limit::Accumulator acc;
    human_limit::initAccumulator(net, pos, &acc);

    std::mt19937 rng(seed);
    constexpr double kTol = 1e-3;

    for (int ply = 0; ply < plies; ply++) {
        auto legal = pos.getValidMoves();
        if (legal.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
        auto m = legal[pick(rng)];
        char promo = (m.promotion == ' ') ? 'q' : m.promotion;

        auto undo = pos.makeMove(m.from, m.to, promo);
        human_limit::applyMoveToAccumulator(net, pos, undo, acc, &acc);

        human_limit::Accumulator fromScratch;
        human_limit::initAccumulator(net, pos, &fromScratch);

        double dw = maxAbsDiff(acc.white, fromScratch.white);
        double db = maxAbsDiff(acc.black, fromScratch.black);
        bool pieceCountOk = acc.pieceCount == fromScratch.pieceCount;
        check(dw < kTol && db < kTol && pieceCountOk,
              label + " ply " + std::to_string(ply) + " diff(w,b)=" + std::to_string(dw) + "," + std::to_string(db) +
                  " pieceCount=" + std::to_string(acc.pieceCount) + "/" + std::to_string(fromScratch.pieceCount));
        if (dw >= kTol || db >= kTol || !pieceCountOk) return;
    }
}

}

int main() {
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    human_limit::Network net;
    bool loaded = net.load("engine/human_limit/nnue_weights.bin");
    check(loaded, "weights load for bitboard accumulator test");
    if (!loaded) {
        std::cout << "TESTS FAILED\n";
        return 1;
    }

    runRandomWalk(net, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 80, 1, "startpos");
    runRandomWalk(net, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 80, 2, "startpos-seed2");
    runRandomWalk(net, "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", 60, 3, "castling-rich");
    runRandomWalk(net, "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 60, 4, "en-passant-classic");
    runRandomWalk(net, "8/P7/k7/8/8/8/7p/K7 w - - 0 1", 40, 5, "promotion-race");
    runRandomWalk(net, "8/8/8/8/8/8/8/K6k w - - 0 1", 20, 6, "bare-kings");

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
