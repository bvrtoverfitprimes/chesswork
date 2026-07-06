#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <iostream>
#include <sstream>

#include "chess/board.h"
#include "engine/search.h"

namespace {

constexpr int kSearchDepth = 4;
constexpr int kMaxPlies = 300;

char pieceLetter(char pieceLower) {
    switch (pieceLower) {
        case 'n': return 'N';
        case 'b': return 'B';
        case 'r': return 'R';
        case 'q': return 'Q';
        case 'k': return 'K';
        default: return '\0';
    }
}

std::string sanForMove(chess::Game& game, const std::string& uci, const std::vector<std::string>& legalMoves) {
    chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
    chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
    char promo = (uci.size() == 5) ? uci[4] : '\0';

    const auto& board = game.boardArray();
    char piece = board[from.r][from.c];
    char pieceLower = std::tolower(static_cast<unsigned char>(piece));

    if (pieceLower == 'k' && std::abs(to.c - from.c) == 2) {
        std::string san = (to.c > from.c) ? "O-O" : "O-O-O";

        auto undo = game.makeMove(from, to, promo ? promo : 'q');
        bool givesCheck = game.inCheck();
        bool noReplies = game.getValidMovesUci().empty();
        game.unmakeMove(undo);

        if (givesCheck) san += noReplies ? "#" : "+";
        return san;
    }

    bool isCapture = board[to.r][to.c] != ' ';
    bool isEnPassant = (pieceLower == 'p') && (from.c != to.c) && !isCapture;
    if (isEnPassant) isCapture = true;

    int sameTypeOtherOrigin = 0;
    bool sameFile = false;
    bool sameRank = false;
    for (const auto& other : legalMoves) {
        if (other.substr(2, 2) != uci.substr(2, 2)) continue;
        chess::Pos otherFrom = chess::Game::parseSquare(other.substr(0, 2));
        if (otherFrom.r == from.r && otherFrom.c == from.c) continue;
        char otherPiece = board[otherFrom.r][otherFrom.c];
        if (std::tolower(static_cast<unsigned char>(otherPiece)) != pieceLower) continue;

        sameTypeOtherOrigin++;
        if (otherFrom.c == from.c) sameFile = true;
        if (otherFrom.r == from.r) sameRank = true;
    }

    std::string san;
    char letter = pieceLetter(pieceLower);
    if (letter) {
        san += letter;
        if (sameTypeOtherOrigin > 0) {
            if (!sameFile) san += chess::LATERAL[from.c];
            else if (!sameRank) san += std::to_string(8 - from.r);
            else san += chess::Game::squareToStr(from);
        }
    } else if (isCapture) {
        san += chess::LATERAL[from.c];
    }

    if (isCapture) san += 'x';
    san += chess::Game::squareToStr(to);
    if (promo) {
        san += '=';
        san += pieceLetter(std::tolower(static_cast<unsigned char>(promo)));
    }

    auto undo = game.makeMove(from, to, promo ? promo : 'q');
    bool givesCheck = game.inCheck();
    bool noReplies = game.getValidMovesUci().empty();
    game.unmakeMove(undo);

    if (givesCheck) san += noReplies ? "#" : "+";
    return san;
}

std::string currentDateForPgn() {
    std::time_t now = std::time(nullptr);
    std::tm* tmPtr = std::gmtime(&now);
    std::ostringstream oss;
    oss << (tmPtr->tm_year + 1900) << "." << (tmPtr->tm_mon + 1) << "." << tmPtr->tm_mday;
    return oss.str();
}

}

int main() {
    chess::Game game;
    std::vector<std::string> sanMoves;
    std::string result = "*";

    for (int ply = 0; ply < kMaxPlies; ply++) {
        if (game.isFiftyMoveDraw() || game.isRepetitionDraw() || game.isInsufficientMaterial()) {
            result = "1/2-1/2";
            break;
        }

        auto legal = game.getValidMovesUci();
        if (legal.empty()) {
            if (game.inCheck()) {
                result = (game.turn() == chess::Color::White) ? "0-1" : "1-0";
            } else {
                result = "1/2-1/2";
            }
            break;
        }

        auto best = engine::findBestMove(game, kSearchDepth);
        std::string san = sanForMove(game, best.uci, legal);
        sanMoves.push_back(san);

        chess::Pos from = chess::Game::parseSquare(best.uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(best.uci.substr(2, 2));
        char promo = (best.uci.size() == 5) ? best.uci[4] : 'q';
        game.makeMove(from, to, promo);
    }

    std::cout << "[Event \"Self-play\"]\n";
    std::cout << "[Site \"?\"]\n";
    std::cout << "[Date \"" << currentDateForPgn() << "\"]\n";
    std::cout << "[Round \"1\"]\n";
    std::cout << "[White \"SimpleEngine\"]\n";
    std::cout << "[Black \"SimpleEngine\"]\n";
    std::cout << "[Result \"" << result << "\"]\n\n";

    for (size_t i = 0; i < sanMoves.size(); i++) {
        if (i % 2 == 0) std::cout << (i / 2 + 1) << ". ";
        std::cout << sanMoves[i] << " ";
    }
    std::cout << result << "\n";

    return 0;
}
