#include "search.h"

#include <algorithm>
#include <cctype>

namespace human_limit {

namespace {

constexpr int kInfinity = 10'000'000;
constexpr int kMateScore = 1'000'000;

int pieceValue(char pieceLower) {
    switch (pieceLower) {
        case 'p': return 100;
        case 'n': return 320;
        case 'b': return 330;
        case 'r': return 500;
        case 'q': return 900;
        default: return 0;
    }
}

bool hasNonPawnMaterial(chess::Game& game) {
    const auto& board = game.boardArray();
    bool white = game.turn() == chess::Color::White;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;
            bool isWhitePiece = std::isupper(static_cast<unsigned char>(p));
            if (isWhitePiece != white) continue;
            char pl = std::tolower(static_cast<unsigned char>(p));
            if (pl != 'p' && pl != 'k') return true;
        }
    }
    return false;
}

}

Searcher::Searcher(const Network& net) : net_(net) {}

bool Searcher::timeExpired() {
    if (timeLimitMs_ <= 0) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startTime_)
                       .count();
    return elapsed >= timeLimitMs_;
}

std::vector<std::string> Searcher::orderMoves(chess::Game& game, const std::vector<std::string>& moves,
                                               int ply, const std::string& ttMove) {
    int clampedPly = std::min(ply, kMaxPly - 1);
    const auto& board = game.boardArray();

    auto scoreOf = [&](const std::string& uci) {
        if (uci == ttMove) return 1'000'000;

        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char captured = board[to.r][to.c];
        if (captured != ' ') {
            chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
            char attacker = board[from.r][from.c];
            return 100'000 + pieceValue(std::tolower(static_cast<unsigned char>(captured))) * 10 -
                   pieceValue(std::tolower(static_cast<unsigned char>(attacker)));
        }
        if (uci == killers_[clampedPly][0]) return 90'000;
        if (uci == killers_[clampedPly][1]) return 89'000;

        auto it = history_.find(uci);
        return it != history_.end() ? it->second : 0;
    };

    std::vector<std::pair<int, std::string>> scored;
    scored.reserve(moves.size());
    for (const auto& m : moves) scored.push_back({scoreOf(m), m});

    std::stable_sort(scored.begin(), scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<std::string> ordered;
    ordered.reserve(scored.size());
    for (auto& [score, uci] : scored) ordered.push_back(uci);
    return ordered;
}

int Searcher::quiescence(chess::Game& game, int alpha, int beta) {
    if (stopped_) return 0;
    nodeCount_++;
    if ((nodeCount_ & 2047) == 0 && timeExpired()) {
        stopped_ = true;
        return 0;
    }

    bool inCheck = game.inCheck();
    int standPat = static_cast<int>(net_.evaluate(game));
    int score = (game.turn() == chess::Color::White) ? standPat : -standPat;

    if (!inCheck) {
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    auto moves = game.getValidMovesUci();
    if (inCheck && moves.empty()) return -kMateScore;

    std::vector<std::string> candidates;
    if (inCheck) {
        candidates = moves;
    } else {
        const auto& board = game.boardArray();
        for (const auto& m : moves) {
            chess::Pos to = chess::Game::parseSquare(m.substr(2, 2));
            if (board[to.r][to.c] != ' ') candidates.push_back(m);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](const std::string& a, const std::string& b) {
        const auto& board = game.boardArray();
        chess::Pos toA = chess::Game::parseSquare(a.substr(2, 2));
        chess::Pos toB = chess::Game::parseSquare(b.substr(2, 2));
        return pieceValue(std::tolower(static_cast<unsigned char>(board[toA.r][toA.c]))) >
               pieceValue(std::tolower(static_cast<unsigned char>(board[toB.r][toB.c])));
    });

    for (const auto& uci : candidates) {
        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';

        auto undo = game.makeMove(from, to, promo);
        int childScore = -quiescence(game, -beta, -alpha);
        game.unmakeMove(undo);

        if (stopped_) return 0;
        if (childScore >= beta) return beta;
        if (childScore > alpha) alpha = childScore;
    }

    return alpha;
}

int Searcher::negamax(chess::Game& game, int depth, int ply, int alpha, int beta, bool allowNull) {
    if (stopped_) return 0;
    nodeCount_++;
    if ((nodeCount_ & 2047) == 0 && timeExpired()) {
        stopped_ = true;
        return 0;
    }

    if (ply >= kMaxPly - 1) {
        int whiteScore = static_cast<int>(net_.evaluate(game));
        return (game.turn() == chess::Color::White) ? whiteScore : -whiteScore;
    }

    if (ply > 0 && (game.isFiftyMoveDraw() || game.isRepetitionDraw() || game.isInsufficientMaterial())) {
        return 0;
    }

    uint64_t key = game.zobristHash();
    size_t idx = key & (kTTSize - 1);
    int alphaOrig = alpha;
    std::string ttMove;

    if (tt_[idx].key == key) {
        ttMove = tt_[idx].bestMove;
        if (tt_[idx].depth >= depth) {
            if (tt_[idx].flag == 0) {
                return tt_[idx].score;
            } else if (tt_[idx].flag == 1) {
                alpha = std::max(alpha, tt_[idx].score);
            } else if (tt_[idx].flag == 2) {
                beta = std::min(beta, tt_[idx].score);
            }
            if (alpha >= beta) return tt_[idx].score;
        }
    }

    bool inCheck = game.inCheck();
    auto moves = game.getValidMovesUci();
    if (moves.empty()) {
        if (inCheck) return -(kMateScore - ply);
        return 0;
    }

    if (depth <= 0) {
        return quiescence(game, alpha, beta);
    }

    if (allowNull && !inCheck && depth >= 3 && hasNonPawnMaterial(game)) {
        auto nullUndo = game.makeNullMove();
        constexpr int kNullReduction = 2;
        int score = -negamax(game, depth - 1 - kNullReduction, ply + 1, -beta, -beta + 1, false);
        game.unmakeNullMove(nullUndo);
        if (stopped_) return 0;
        if (score >= beta) return beta;
    }

    auto ordered = orderMoves(game, moves, ply, ttMove);

    int best = -kInfinity;
    std::string bestMove;
    int moveIndex = 0;

    for (const auto& uci : ordered) {
        chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
        char promo = (uci.size() == 5) ? uci[4] : 'q';
        bool isCapture = game.boardArray()[to.r][to.c] != ' ';
        bool isPromotion = uci.size() == 5;

        auto undo = game.makeMove(from, to, promo);

        int score;
        if (moveIndex >= 3 && depth >= 3 && !isCapture && !isPromotion && !game.inCheck()) {
            score = -negamax(game, depth - 2, ply + 1, -alpha - 1, -alpha, true);
            if (score > alpha && !stopped_) {
                score = -negamax(game, depth - 1, ply + 1, -beta, -alpha, true);
            }
        } else {
            score = -negamax(game, depth - 1, ply + 1, -beta, -alpha, true);
        }

        game.unmakeMove(undo);
        if (stopped_) return 0;

        if (score > best) {
            best = score;
            bestMove = uci;
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) {
            if (!isCapture) {
                int clampedPly = std::min(ply, kMaxPly - 1);
                if (killers_[clampedPly][0] != uci) {
                    killers_[clampedPly][1] = killers_[clampedPly][0];
                    killers_[clampedPly][0] = uci;
                }
                history_[uci] += depth * depth;
            }
            break;
        }
        moveIndex++;
    }

    int flag = (best <= alphaOrig) ? 2 : (best >= beta ? 1 : 0);
    tt_[idx] = {key, depth, best, flag, bestMove};

    return best;
}

SearchResult Searcher::findBestMove(chess::Game& game, int maxDepth, int timeMs) {
    startTime_ = std::chrono::steady_clock::now();
    timeLimitMs_ = timeMs;
    stopped_ = false;
    nodeCount_ = 0;

    SearchResult result;
    int prevScore = 0;

    auto rootMoves = game.getValidMovesUci();
    if (rootMoves.empty()) return result;
    result.uci = rootMoves.front();

    for (int depth = 1; depth <= maxDepth; depth++) {
        int windowAlpha = (depth >= 4) ? prevScore - 50 : -kInfinity;
        int windowBeta = (depth >= 4) ? prevScore + 50 : kInfinity;

        std::string bestMoveThisDepth;
        int bestScoreThisDepth = -kInfinity;
        bool aborted = false;

        while (true) {
            bestMoveThisDepth.clear();
            bestScoreThisDepth = -kInfinity;
            int a = windowAlpha;
            int b = windowBeta;

            auto ordered = orderMoves(game, rootMoves, 0, result.uci);
            for (const auto& uci : ordered) {
                chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
                chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
                char promo = (uci.size() == 5) ? uci[4] : 'q';

                auto undo = game.makeMove(from, to, promo);
                int score = -negamax(game, depth - 1, 1, -b, -a, true);
                game.unmakeMove(undo);

                if (stopped_) {
                    aborted = true;
                    break;
                }
                if (score > bestScoreThisDepth) {
                    bestScoreThisDepth = score;
                    bestMoveThisDepth = uci;
                }
                if (score > a) a = score;
            }

            if (aborted) break;
            if (bestScoreThisDepth <= windowAlpha || bestScoreThisDepth >= windowBeta) {
                windowAlpha = -kInfinity;
                windowBeta = kInfinity;
                continue;
            }
            break;
        }

        if (aborted) break;

        result.uci = bestMoveThisDepth;
        result.score = bestScoreThisDepth;
        result.depthReached = depth;
        prevScore = bestScoreThisDepth;

        if (timeExpired()) break;
    }

    result.nodes = nodeCount_;
    return result;
}

}
