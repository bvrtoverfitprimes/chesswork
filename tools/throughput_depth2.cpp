#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../chess/board.h"
#include "../engine/human_limit/network.h"
#include "../engine/human_limit/search.h"
#include "../engine/old_engine/search.h"

int main(int argc, char** argv) {
    int iterations = argc > 1 ? std::atoi(argv[1]) : 500;
    int depth = argc > 2 ? std::atoi(argv[2]) : 2;

    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    human_limit::Network net;
    net.load("engine/human_limit/nnue_weights.bin");

    std::mt19937 rng(42);

    {
        human_limit::Searcher searcher(net);
        long totalNodes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            chess::bitboard::Position pos;
            std::uniform_int_distribution<int> plyDist(0, 30);
            int plies = plyDist(rng);
            for (int p = 0; p < plies; p++) {
                auto legal = pos.getValidMoves();
                if (legal.empty()) break;
                std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
                auto m = legal[pick(rng)];
                char promo = (m.promotion == ' ') ? 'q' : m.promotion;
                pos.makeMove(m.from, m.to, promo);
            }
            auto r = searcher.findBestMove(pos, depth, 60000);
            totalNodes += r.nodes;
        }
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        double posPerSec = iterations * 1000.0 / std::max<long long>(elapsedMs, 1);
        std::cout << "human_limit depth=" << depth << ": " << iterations << " positions in "
                  << elapsedMs << "ms -> " << posPerSec << " pos/sec, "
                  << (double(elapsedMs) / iterations) << "ms/pos, avg_nodes="
                  << (totalNodes / iterations) << "\n";
    }

    rng.seed(42);
    {
        old_engine::Searcher searcher;
        long totalNodes = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            chess::Game game;
            std::uniform_int_distribution<int> plyDist(0, 30);
            int plies = plyDist(rng);
            for (int p = 0; p < plies; p++) {
                auto legal = game.getValidMovesUci();
                if (legal.empty()) break;
                std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
                std::string uci = legal[pick(rng)];
                chess::Pos from = chess::Game::parseSquare(uci.substr(0, 2));
                chess::Pos to = chess::Game::parseSquare(uci.substr(2, 2));
                char promo = (uci.size() == 5) ? uci[4] : 'q';
                game.makeMove(from, to, promo);
            }
            auto r = searcher.findBestMove(game, depth, 60000);
            totalNodes += r.nodes;
        }
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        double posPerSec = iterations * 1000.0 / std::max<long long>(elapsedMs, 1);
        std::cout << "old_engine  depth=" << depth << ": " << iterations << " positions in "
                  << elapsedMs << "ms -> " << posPerSec << " pos/sec, "
                  << (double(elapsedMs) / iterations) << "ms/pos, avg_nodes="
                  << (totalNodes / iterations) << "\n";
    }

    return 0;
}
