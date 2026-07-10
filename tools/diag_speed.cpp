#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../chess/board.h"
#include "../engine/limit/network.h"
#include "../engine/limit/search.h"
#include "../engine/old_engine/search.h"

int main(int argc, char** argv) {
    std::string fen = argc > 1 ? argv[1] : "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    int timeMs = argc > 2 ? std::atoi(argv[2]) : 300;
    int maxDepth = argc > 3 ? std::atoi(argv[3]) : 64;

    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    limit::Network net;
    net.load("engine/limit/nnue_weights.bin");

    {
        chess::bitboard::Position pos(fen);
        limit::Searcher searcher(net);
        auto t0 = std::chrono::steady_clock::now();
        auto r = searcher.findBestMove(pos, maxDepth, timeMs);
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "limit: move=" << r.uci << " depth=" << r.depthReached
                  << " nodes=" << r.nodes << " score=" << r.score << " elapsedMs=" << elapsedMs
                  << " nps=" << (r.nodes * 1000 / std::max<long long>(elapsedMs, 1)) << "\n";
    }
    {
        chess::Game game(fen);
        old_engine::Searcher searcher;
        auto t0 = std::chrono::steady_clock::now();
        auto r = searcher.findBestMove(game, maxDepth, timeMs);
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "old_engine:  move=" << r.uci << " depth=" << r.depthReached
                  << " nodes=" << r.nodes << " score=" << r.score << " elapsedMs=" << elapsedMs
                  << " nps=" << (r.nodes * 1000 / std::max<long long>(elapsedMs, 1)) << "\n";
    }
    return 0;
}
