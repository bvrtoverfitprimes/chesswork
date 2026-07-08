#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../engine/human_limit/accumulator.h"
#include "../engine/human_limit/network.h"
#include "../engine/human_limit/nnue_features.h"

using namespace chess::bitboard;

int main(int argc, char** argv) {
    int iterations = argc > 1 ? std::atoi(argv[1]) : 20000;

    initAttackTables();
    initMagics();

    human_limit::Network net;
    net.load("engine/human_limit/nnue_weights.bin");

    std::mt19937 rng(7);
    std::vector<Position> positions;
    for (int i = 0; i < 2000; i++) {
        Position pos;
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
        positions.push_back(pos);
    }

    // 1. Pure move generation cost.
    {
        auto t0 = std::chrono::steady_clock::now();
        volatile size_t sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto& pos = positions[i % positions.size()];
            auto moves = pos.getValidMoves();
            sink += moves.size();
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "getValidMoves() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call (sink=" << sink << ")\n";
    }

    // 2. Pure NNUE eval cost (full recompute, matching what happens once per findBestMove root init).
    {
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto& pos = positions[i % positions.size()];
            human_limit::Accumulator acc;
            human_limit::initAccumulator(net, pos, &acc);
            int bucket = human_limit::outputBucketFromPieceCount(acc.pieceCount);
            sink += net.evaluateFromAccumulators(acc.white, acc.black, pos.turn(), bucket);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "full NNUE eval (accumulator init + head): " << iterations << " calls in " << elapsedUs
                  << "us -> " << (double(elapsedUs) / iterations) << "us/call (sink=" << sink << ")\n";
    }

    // 2b. Just the accumulator full-recompute (embedding sum), no head.
    {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            auto& pos = positions[i % positions.size()];
            human_limit::Accumulator acc;
            human_limit::initAccumulator(net, pos, &acc);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "accumulator full recompute only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    // 2c. Just the head forward pass, given a precomputed accumulator (the incremental-update steady state).
    {
        auto& pos0 = positions[0];
        human_limit::Accumulator acc;
        human_limit::initAccumulator(net, pos0, &acc);
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0;
        for (int i = 0; i < iterations; i++) {
            int bucket = human_limit::outputBucketFromPieceCount(acc.pieceCount);
            sink += net.evaluateFromAccumulators(acc.white, acc.black, pos0.turn(), bucket);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "head forward pass only (given accumulator): " << iterations << " calls in " << elapsedUs
                  << "us -> " << (double(elapsedUs) / iterations) << "us/call (sink=" << sink << ")\n";
    }

    // 3. makeMove/unmakeMove cost alone.
    {
        auto& pos = positions[0];
        auto moves = pos.getValidMoves();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            auto& m = moves[i % moves.size()];
            char promo = (m.promotion == ' ') ? 'q' : m.promotion;
            auto undo = pos.makeMove(m.from, m.to, promo);
            pos.unmakeMove(undo);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "makeMove+unmakeMove only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    // 4. Incremental accumulator update cost alone (the steady-state search cost, not full recompute).
    {
        auto& pos = positions[0];
        auto moves = pos.getValidMoves();
        human_limit::Accumulator acc;
        human_limit::initAccumulator(net, pos, &acc);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            auto& m = moves[i % moves.size()];
            char promo = (m.promotion == ' ') ? 'q' : m.promotion;
            auto undo = pos.makeMove(m.from, m.to, promo);
            human_limit::Accumulator next;
            human_limit::applyMoveToAccumulator(net, pos, undo, acc, &next);
            pos.unmakeMove(undo);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "incremental accumulator update only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    return 0;
}
