#include <algorithm>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../chess/board.h"

namespace {

int failures = 0;
int positionsChecked = 0;

void checkPosition(chess::Game& game, const std::string& label) {
    auto fastMoves = game.getValidMovesUci();
    auto slowMoves = game.getValidMovesUciSlow();

    std::set<std::string> fastSet(fastMoves.begin(), fastMoves.end());
    std::set<std::string> slowSet(slowMoves.begin(), slowMoves.end());

    if (fastSet != slowSet) {
        std::cout << "FAIL: " << label << " fast(" << fastSet.size() << ") != slow(" << slowSet.size() << ")\n";
        std::cout << "  fen: " << game.toFen() << "\n";
        for (const auto& m : fastSet) if (!slowSet.count(m)) std::cout << "  extra in fast: " << m << "\n";
        for (const auto& m : slowSet) if (!fastSet.count(m)) std::cout << "  missing from fast: " << m << "\n";
        failures++;
    } else {
        positionsChecked++;
    }
}

void randomWalk(const std::string& fen, int plies, uint32_t seed, const std::string& label) {
    chess::Game game(fen);
    std::mt19937 rng(seed);
    for (int ply = 0; ply < plies; ply++) {
        checkPosition(game, label + " ply " + std::to_string(ply));
        auto legal = game.getValidMovesUciSlow();
        if (legal.empty()) break;
        std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
        std::string uci = legal[pick(rng)];
        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';
        game.makeMove(from, to, promo);
    }
}

}

int main() {
    for (uint32_t seed = 1; seed <= 30; seed++) {
        randomWalk("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 80, seed, "startpos-seed" + std::to_string(seed));
    }

    randomWalk("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", 60, 101, "castling-rich");
    randomWalk("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 60, 102, "en-passant-classic");
    randomWalk("8/P7/k7/8/8/8/7p/K7 w - - 0 1", 40, 103, "promotion-race");

    struct Fen {
        std::string fen;
        std::string label;
    };
    std::vector<Fen> tricky = {
        {"6k1/8/8/8/1r6/8/3K4/8 w - - 0 1", "rook-pin-diagonal-none"},
        {"6k1/8/8/8/8/8/3K1r2/3Q4 w - - 0 1", "queen-pinned-by-rook-same-rank"},
        {"3r2k1/8/8/8/8/8/3Q4/3K4 w - - 0 1", "queen-pinned-vertical"},
        {"6k1/8/8/8/8/2b5/3N4/4K3 w - - 0 1", "knight-pinned-diagonal"},
        {"8/8/8/8/8/2k5/8/K1n1R3 w - - 0 1", "no-pin-check-adjacent"},
        {"4k3/8/8/8/8/4r3/8/3RK3 w - - 0 1", "king-in-check-by-rook-same-file"},
        {"4k3/8/8/8/8/8/4n3/4K3 w - - 0 1", "king-in-check-by-knight"},
        {"4k3/8/8/8/8/2n5/8/4K3 w - - 0 1", "knight-check-block-impossible"},
        {"4k3/8/8/3b4/8/8/8/4K3 w - - 0 1", "no-check-far-bishop"},
        {"1k1r4/8/8/8/8/8/8/R2K3r w - - 0 1", "double-check-candidate"},
        {"7k/8/8/8/8/8/8/RN2K2r w - - 0 1", "rook-and-knight-both-attacking"},
        {"4k3/8/8/8/4r3/8/4P3/4K3 w - - 0 1", "pinned-pawn-vertical"},
        {"4k3/8/8/8/8/8/8/2r1K2R w K - 0 1", "rook-pinned-would-block-castle-path"},
        {"rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "en-passant-available"},
        {"8/8/1k6/8/2pP4/8/8/2K3R1 b - d3 0 1", "en-passant-discovered-check-setup"},
    };
    for (auto& t : tricky) {
        chess::Game game(t.fen);
        checkPosition(game, t.label);
        randomWalk(t.fen, 30, 200 + static_cast<uint32_t>(&t - &tricky[0]), t.label + "-walk");
    }

    std::cout << positionsChecked << " positions checked, " << failures << " mismatches\n";
    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
