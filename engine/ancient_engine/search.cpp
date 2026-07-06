#include "search.h"

#include <algorithm>

#include "evaluation.h"

namespace ancient_engine {

namespace {

constexpr int kInfinity = 10'000'000;
constexpr int kMateScore = 1'000'000;

std::vector<std::string> orderedMoves(chess::Game& game, const std::vector<std::string>& moves) {
    std::vector<std::string> ordered = moves;
    const auto& board = game.boardArray();
    std::stable_sort(ordered.begin(), ordered.end(), [&](const std::string& a, const std::string& b) {
        chess::Pos toA = chess::Game::parseSquare(a.substr(2, 2));
        chess::Pos toB = chess::Game::parseSquare(b.substr(2, 2));
        bool captureA = board[toA.r][toA.c] != ' ';
        bool captureB = board[toB.r][toB.c] != ' ';
        return captureA > captureB;
    });
    return ordered;
}

int negamax(chess::Game& game, int depth, int ply, int alpha, int beta) {
    if (game.isFiftyMoveDraw() || game.isRepetitionDraw() || game.isInsufficientMaterial()) {
        return 0;
    }

    auto moves = game.getValidMovesUci();
    if (moves.empty()) {
        if (game.inCheck()) return -(kMateScore - ply);
        return 0;
    }

    if (depth == 0) {
        int whiteScore = evaluate(game.boardArray());
        return (game.turn() == chess::Color::White) ? whiteScore : -whiteScore;
    }

    int best = -kInfinity;
    for (const auto& uci : orderedMoves(game, moves)) {
        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';

        auto undo = game.makeMove(from, to, promo);
        int score = -negamax(game, depth - 1, ply + 1, -beta, -alpha);
        game.unmakeMove(undo);

        if (score > best) best = score;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }
    return best;
}

}

SearchResult findBestMove(chess::Game& game, int depth) {
    auto moves = game.getValidMovesUci();
    SearchResult result{"", -kInfinity};
    if (moves.empty()) return result;

    int alpha = -kInfinity;
    int beta = kInfinity;

    for (const auto& uci : orderedMoves(game, moves)) {
        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';

        auto undo = game.makeMove(from, to, promo);
        int score = -negamax(game, depth - 1, 1, -beta, -alpha);
        game.unmakeMove(undo);

        if (score > result.score) {
            result.score = score;
            result.uci = uci;
        }
        if (score > alpha) alpha = score;
    }
    return result;
}

}
