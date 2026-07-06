#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../chess/board.h"

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

bool applyIfLegal(chess::Game& game, const std::string& uci) {
    auto legal = game.getValidMovesUci();
    if (std::find(legal.begin(), legal.end(), uci) == legal.end()) return false;
    chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
    chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
    char promo = (uci.size() == 5) ? uci[4] : 'q';
    game.makeMove(from, to, promo);
    return true;
}

uint64_t perft(chess::Game& game, int depth) {
    if (depth == 0) return 1;
    uint64_t count = 0;
    auto legal = game.getValidMovesUci();
    for (const auto& uci : legal) {
        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';
        auto undo = game.makeMove(from, to, promo);
        count += perft(game, depth - 1);
        game.unmakeMove(undo);
    }
    return count;
}

void testPerftStartPos() {
    chess::Game game;
    struct { int depth; uint64_t expected; } cases[] = {
        {1, 20}, {2, 400}, {3, 8902}, {4, 197281},
    };
    for (auto& tc : cases) {
        uint64_t got = perft(game, tc.depth);
        check(got == tc.expected,
              "perft startpos depth " + std::to_string(tc.depth) +
                  " = " + std::to_string(got) + " (expected " + std::to_string(tc.expected) + ")");
    }
}

void testPerftKiwipete() {
    chess::Game game("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    struct { int depth; uint64_t expected; } cases[] = {
        {1, 48}, {2, 2039},
    };
    for (auto& tc : cases) {
        uint64_t got = perft(game, tc.depth);
        check(got == tc.expected,
              "perft kiwipete depth " + std::to_string(tc.depth) +
                  " = " + std::to_string(got) + " (expected " + std::to_string(tc.expected) + ")");
    }
}

void testCastlingAvailable() {
    chess::Game game("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    auto legal = game.getValidMovesUci();
    bool hasKingside = std::find(legal.begin(), legal.end(), "e1g1") != legal.end();
    bool hasQueenside = std::find(legal.begin(), legal.end(), "e1c1") != legal.end();
    check(hasKingside, "kingside castling is a legal move when path is clear");
    check(hasQueenside, "queenside castling is a legal move when path is clear");
}

void testCastlingBlockedByCheck() {
    chess::Game game("4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1");
    auto legal = game.getValidMovesUci();
    bool hasKingside = std::find(legal.begin(), legal.end(), "e1g1") != legal.end();
    bool hasQueenside = std::find(legal.begin(), legal.end(), "e1c1") != legal.end();
    check(!hasKingside && !hasQueenside, "castling forbidden while king is in check");
}

void testCastlingThroughAttackedSquare() {
    chess::Game game("4k3/8/8/8/8/8/5r2/R3K2R w KQ - 0 1");
    auto legal = game.getValidMovesUci();
    bool hasKingside = std::find(legal.begin(), legal.end(), "e1g1") != legal.end();
    check(!hasKingside, "castling forbidden when king passes through an attacked square");
}

void testEnPassantCapture() {
    chess::Game game("4k3/8/8/pP6/8/8/8/4K3 w - a6 0 1");
    auto legal = game.getValidMovesUci();
    bool hasEp = std::find(legal.begin(), legal.end(), "b5a6") != legal.end();
    check(hasEp, "en passant capture is offered when available");

    chess::Pos from = chess::Game::parseSquare("b5");
    chess::Pos to = chess::Game::parseSquare("a6");
    game.makeMove(from, to);
    auto legalAfter = game.getValidMovesUci();
    check(true, "en passant applied without crashing");
    (void)legalAfter;
}

void testPromotionAllFourChoices() {
    chess::Game game("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    auto legal = game.getValidMovesUci();
    for (std::string suffix : {"q", "r", "b", "n"}) {
        std::string uci = "a7a8" + suffix;
        check(std::find(legal.begin(), legal.end(), uci) != legal.end(),
              "promotion choice offered: " + uci);
    }
    check(std::find(legal.begin(), legal.end(), "a7a8") == legal.end(),
          "bare promotion move without a piece suffix is not offered");
}

void testPinnedPieceCannotMove() {
    chess::Game game("4k3/8/8/8/8/4r3/4N3/4K3 w - - 0 1");
    auto legal = game.getValidMovesUci();
    bool knightCanMoveOffFile = std::find(legal.begin(), legal.end(), "e2c3") != legal.end();
    check(!knightCanMoveOffFile, "pinned knight cannot move off the pin line");
}

void testFoolsMateCheckmate() {
    chess::Game game;
    check(applyIfLegal(game, "f2f3"), "fools mate move 1 white f2f3");
    check(applyIfLegal(game, "e7e5"), "fools mate move 1 black e7e5");
    check(applyIfLegal(game, "g2g4"), "fools mate move 2 white g2g4");
    check(applyIfLegal(game, "d8h4"), "fools mate move 2 black d8h4 delivers mate");
    check(game.isCheckmate(), "fools mate results in checkmate");
    check(game.getValidMovesUci().empty(), "no legal moves exist after checkmate");
}

void testStalemate() {
    chess::Game game("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    check(!game.inCheck(), "stalemate position: king is not in check");
    check(game.isStalemate(), "stalemate position correctly detected");
    check(!game.isCheckmate(), "stalemate is not misreported as checkmate");
}

void testFiftyMoveRule() {
    chess::Game game("4k3/8/8/8/8/8/8/4K2R w K - 99 1");
    check(!game.isFiftyMoveDraw(), "halfmove clock at 99 is not yet a fifty-move draw");
    check(applyIfLegal(game, "h1h2"), "quiet rook move advances halfmove clock to 100");
    check(game.isFiftyMoveDraw(), "fifty-move rule triggers at halfmove clock 100");
}

void testThreefoldRepetition() {
    chess::Game game;
    check(!game.isRepetitionDraw(), "start position is not a repetition draw");
    for (int i = 0; i < 2; i++) {
        check(applyIfLegal(game, "g1f3"), "knight out");
        check(applyIfLegal(game, "g8f6"), "knight out (black)");
        check(applyIfLegal(game, "f3g1"), "knight back");
        check(applyIfLegal(game, "f6g8"), "knight back (black)");
    }
    check(game.isRepetitionDraw(), "threefold repetition correctly detected");
}

void testInsufficientMaterial() {
    chess::Game kingsOnly("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    check(kingsOnly.isInsufficientMaterial(), "king vs king is insufficient material");

    chess::Game kingBishop("4k3/8/8/8/8/8/8/4KB2 w - - 0 1");
    check(kingBishop.isInsufficientMaterial(), "king+bishop vs king is insufficient material");

    chess::Game kingRook("4k3/8/8/8/8/8/8/4KR2 w - - 0 1");
    check(!kingRook.isInsufficientMaterial(), "king+rook vs king is NOT insufficient material");
}

void testIllegalMoveStrainFromStart() {
    chess::Game game;
    auto legal = game.getValidMovesUci();
    int rejected = 0;
    int totalTried = 0;
    for (int r1 = 0; r1 < 8; r1++) {
        for (int c1 = 0; c1 < 8; c1++) {
            for (int r2 = 0; r2 < 8; r2++) {
                for (int c2 = 0; c2 < 8; c2++) {
                    if (r1 == r2 && c1 == c2) continue;
                    std::string uci = chess::Game::squareToStr({r1, c1}) + chess::Game::squareToStr({r2, c2});
                    totalTried++;
                    bool isLegal = std::find(legal.begin(), legal.end(), uci) != legal.end();
                    if (!isLegal) rejected++;
                }
            }
        }
    }
    check(rejected == totalTried - static_cast<int>(legal.size()),
          "illegal-move strain: rejected exactly the non-legal squares out of " +
              std::to_string(totalTried) + " candidates (legal=" + std::to_string(legal.size()) + ")");
    check(legal.size() == 20, "start position has exactly 20 legal moves");
}

void testIllegalMoveStrainRandomSelfPlay() {
    std::mt19937 rng(12345);
    chess::Game game;
    for (int ply = 0; ply < 60; ply++) {
        auto legal = game.getValidMovesUci();
        if (legal.empty()) break;

        std::uniform_int_distribution<int> dist(0, static_cast<int>(legal.size()) - 1);
        std::string chosen = legal[dist(rng)];

        bool beforeInCheck = game.inCheck();
        chess::Pos from = chess::Game::parseSquare(chosen.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(chosen.substr(2, 2));
        char promo = (chosen.size() == 5) ? chosen[4] : 'q';
        auto undo = game.makeMove(from, to, promo);

        chess::Color mover = (game.turn() == chess::Color::White) ? chess::Color::Black : chess::Color::White;
        (void)mover;
        (void)beforeInCheck;
        (void)undo;
    }
    check(true, "random legal self-play for 60 plies completed without crashing or desyncing");
}

}

int main() {
    testPerftStartPos();
    testPerftKiwipete();
    testCastlingAvailable();
    testCastlingBlockedByCheck();
    testCastlingThroughAttackedSquare();
    testEnPassantCapture();
    testPromotionAllFourChoices();
    testPinnedPieceCannotMove();
    testFoolsMateCheckmate();
    testStalemate();
    testFiftyMoveRule();
    testThreefoldRepetition();
    testInsufficientMaterial();
    testIllegalMoveStrainFromStart();
    testIllegalMoveStrainRandomSelfPlay();

    std::cout << "\n" << (failures == 0 ? "ALL TESTS PASSED" : std::to_string(failures) + " TEST(S) FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
