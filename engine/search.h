#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "../chess/board.h"

namespace engine {

struct SearchResult {
    std::string uci;
    int score = 0;
    int depthReached = 0;
    long nodes = 0;
};

class Searcher {
public:
    SearchResult findBestMove(chess::Game& game, int maxDepth, int timeMs);

private:
    struct TTEntry {
        uint64_t key = 0;
        int depth = -1;
        int score = 0;
        int flag = 0;
        std::string bestMove;
    };

    static constexpr size_t kTTSize = 1u << 20;
    static constexpr int kMaxPly = 128;

    std::vector<TTEntry> tt_ = std::vector<TTEntry>(kTTSize);
    std::string killers_[kMaxPly][2];
    std::unordered_map<std::string, int> history_;

    std::chrono::steady_clock::time_point startTime_;
    int timeLimitMs_ = 0;
    bool stopped_ = false;
    long nodeCount_ = 0;

    bool timeExpired();
    std::vector<std::string> orderMoves(chess::Game& game, const std::vector<std::string>& moves,
                                         int ply, const std::string& ttMove);
    int quiescence(chess::Game& game, int alpha, int beta);
    int negamax(chess::Game& game, int depth, int ply, int alpha, int beta, bool allowNull);
};

}
