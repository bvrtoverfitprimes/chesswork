#include <algorithm>
#include <iostream>

#include "chess/board.h"
#include "engine/search.h"

namespace {

constexpr int kSearchDepth = 4;

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
    std::cout << "Play as (w)hite or (b)lack? ";
    std::string choice;
    std::cin >> choice;
    chess::Color humanColor = (!choice.empty() && (choice[0] == 'b' || choice[0] == 'B'))
                                   ? chess::Color::Black
                                   : chess::Color::White;

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
            auto result = engine::findBestMove(game, kSearchDepth);
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
