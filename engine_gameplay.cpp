#include <algorithm>
#include <iostream>

#include "chess/bitboard/bitboard.h"
#include "chess/bitboard/magic.h"
#include "chess/bitboard/position.h"
#include "chess/board.h"
#include "engine/limit/network.h"
#include "engine/limit/search.h"

namespace {

constexpr int kSearchDepth = 64;
constexpr int kSearchTimeMs = 2000;
constexpr const char* kWeightsPath = "engine/limit/nnue_weights.bin";

bool isGameOver(chess::Game& game) {
    if (game.isFiftyMoveDraw()) {
        std::cout << "Draw by fifty-move rule.\n";
        return true;
    }
    if (game.isRepetitionDraw()) {
        std::cout << "Draw by repetition.\n";
        return true;
    }
    if (game.isInsufficientMaterial()) {
        std::cout << "Draw by insufficient material.\n";
        return true;
    }
    if (game.getValidMovesUci().empty()) {
        if (game.inCheck()) {
            std::cout << "Checkmate! " << (game.turn() == chess::Color::White ? "Black" : "White") << " wins.\n";
        } else {
            std::cout << "Stalemate!\n";
        }
        return true;
    }
    return false;
}

}

int main() {
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    std::cout << "Play as (w)hite or (b)lack? ";
    std::string choice;
    std::cin >> choice;
    chess::Color humanColor = (!choice.empty() && (choice[0] == 'b' || choice[0] == 'B'))
                                   ? chess::Color::Black
                                   : chess::Color::White;

    limit::Network net;
    if (!net.load(kWeightsPath)) {
        std::cout << "Warning: no trained weights found at " << kWeightsPath
                  << ", playing with an untrained network.\n";
    }
    limit::Searcher searcher(net);

    chess::Game game;
    game.printBoard();

    while (!isGameOver(game)) {
        if (game.turn() == humanColor) {
            auto legal = game.getValidMovesUci();
            std::cout << "\n" << (game.turn() == chess::Color::White ? "white" : "black") << "'s move: ";
            std::string moveInput;
            if (!(std::cin >> moveInput)) break;

            if (std::find(legal.begin(), legal.end(), moveInput) == legal.end()) {
                std::cout << "Invalid move.\n";
                continue;
            }

            chess::Pos from = chess::Game::parseSquare(moveInput.substr(0, 2));
            chess::Pos to = chess::Game::parseSquare(moveInput.substr(2, 2));
            char promo = (moveInput.size() == 5) ? moveInput[4] : 'q';
            game.makeMove(from, to, promo);
        } else {
            chess::bitboard::Position pos(game.toFen());
            auto result = searcher.findBestMove(pos, kSearchDepth, kSearchTimeMs);
            std::cout << "\nEngine plays: " << result.uci << "\n";
            chess::Pos from = chess::Game::parseSquare(result.uci.substr(0, 2));
            chess::Pos to = chess::Game::parseSquare(result.uci.substr(2, 2));
            char promo = (result.uci.size() == 5) ? result.uci[4] : 'q';
            game.makeMove(from, to, promo);
        }
        game.printBoard();
    }

    return 0;
}
