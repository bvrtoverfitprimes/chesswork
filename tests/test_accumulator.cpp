#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../chess/board.h"
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
    chess::Game game(fen);
    human_limit::Accumulator acc;
    human_limit::initAccumulator(net, game.boardArray(), &acc);

    std::mt19937 rng(seed);
    constexpr double kTol = 1e-3;

    for (int ply = 0; ply < plies; ply++) {
        auto legal = game.getValidMovesUci();
        if (legal.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
        std::string uci = legal[pick(rng)];

        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';

        auto undo = game.makeMove(from, to, promo);
        human_limit::applyMoveToAccumulator(net, game.boardArray(), undo, acc, &acc);

        human_limit::Accumulator fromScratch;
        human_limit::initAccumulator(net, game.boardArray(), &fromScratch);

        double dw = maxAbsDiff(acc.white, fromScratch.white);
        double db = maxAbsDiff(acc.black, fromScratch.black);
        check(dw < kTol && db < kTol,
              label + " ply " + std::to_string(ply) + " move " + uci + " diff(w,b)=" +
                  std::to_string(dw) + "," + std::to_string(db));
        if (dw >= kTol || db >= kTol) return;
    }
}

}

int main() {
    human_limit::Network net;
    bool loaded = net.load("engine/human_limit/nnue_weights.bin");
    check(loaded, "weights load for accumulator test");
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
