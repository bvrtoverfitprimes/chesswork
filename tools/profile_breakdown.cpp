#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../engine/limit/accumulator.h"
#include "../engine/limit/network.h"
#include "../engine/limit/nnue_features.h"

using namespace chess::bitboard;

int main(int argc, char** argv) {
    int iterations = argc > 1 ? std::atoi(argv[1]) : 20000;

    initAttackTables();
    initMagics();

    limit::Network net;
    net.load("engine/limit/nnue_weights.bin");

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
            limit::Accumulator acc;
            limit::initAccumulator(net, pos, &acc);
            int bucket = limit::outputBucketFromPieceCount(acc.pieceCount);
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
            limit::Accumulator acc;
            limit::initAccumulator(net, pos, &acc);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "accumulator full recompute only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    // 2c. Just the head forward pass, given a precomputed accumulator (the incremental-update steady state).
    {
        auto& pos0 = positions[0];
        limit::Accumulator acc;
        limit::initAccumulator(net, pos0, &acc);
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0;
        for (int i = 0; i < iterations; i++) {
            int bucket = limit::outputBucketFromPieceCount(acc.pieceCount);
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
        limit::Accumulator acc;
        limit::initAccumulator(net, pos, &acc);
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; i++) {
            auto& m = moves[i % moves.size()];
            char promo = (m.promotion == ' ') ? 'q' : m.promotion;
            auto undo = pos.makeMove(m.from, m.to, promo);
            limit::Accumulator next;
            limit::applyMoveToAccumulator(net, pos, undo, acc, &next);
            pos.unmakeMove(undo);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "incremental accumulator update only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    // 5. pos.toBoardArray() alone (bitboard -> mailbox conversion, redone every eval call).
    {
        auto t0 = std::chrono::steady_clock::now();
        volatile int sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto& pos = positions[i % positions.size()];
            auto board = pos.toBoardArray();
            sink += board[0][0];
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "pos.toBoardArray() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    // 6. computePerspectiveContext() alone (called twice per eval: once per side).
    {
        auto& pos0 = positions[0];
        auto board = pos0.toBoardArray();
        auto t0 = std::chrono::steady_clock::now();
        volatile int sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto ctx = limit::computePerspectiveContext(board, i % 2 == 0);
            sink += ctx.kingBucket;
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "computePerspectiveContext() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    // 7. computeThreatFacts() alone (O(pieces^2) mailbox ray-walk, called once per eval).
    {
        auto& pos0 = positions[0];
        auto board = pos0.toBoardArray();
        auto t0 = std::chrono::steady_clock::now();
        volatile size_t sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto facts = limit::computeThreatFacts(board);
            sink += facts.size();
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "computeThreatFacts() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call (sink=" << sink << ")\n";
    }

    // 7b. computeThreatFactsBB() alone -- bitboard-native replacement, no mailbox ray-walk.
    {
        auto& pos0 = positions[0];
        auto t0 = std::chrono::steady_clock::now();
        volatile size_t sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto facts = limit::computeThreatFactsBB(pos0);
            sink += facts.size();
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "computeThreatFactsBB() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call (sink=" << sink << ")\n";
    }

    // 8. Full evaluateFromAccumulatorsWithThreats() -- the ACTUAL hot path used by Searcher::evalWhiteRelative.
    {
        auto& pos0 = positions[0];
        limit::Accumulator acc;
        limit::initAccumulator(net, pos0, &acc);
        auto t0 = std::chrono::steady_clock::now();
        volatile double sink = 0;
        for (int i = 0; i < iterations; i++) {
            int bucket = limit::outputBucketFromPieceCount(acc.pieceCount);
            sink += net.evaluateFromAccumulatorsWithThreats(acc.white, acc.black, pos0, pos0.turn(), bucket);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "FULL evaluateFromAccumulatorsWithThreats() (actual search hot path): " << iterations
                  << " calls in " << elapsedUs << "us -> " << (double(elapsedUs) / iterations)
                  << "us/call (sink=" << sink << ")\n";
    }

    // 9. inCheck() and see() alone -- pruning-side board queries (already bitboard-based; expected cheap).
    {
        auto& pos = positions[0];
        auto t0 = std::chrono::steady_clock::now();
        volatile bool sink = false;
        for (int i = 0; i < iterations; i++) sink = pos.inCheck();
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "pos.inCheck() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }
    {
        auto& pos = positions[0];
        auto moves = pos.getValidMoves();
        auto t0 = std::chrono::steady_clock::now();
        volatile int sink = 0;
        for (int i = 0; i < iterations; i++) {
            auto& m = moves[i % moves.size()];
            sink += pos.see(m.from, m.to);
        }
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                             std::chrono::steady_clock::now() - t0).count();
        std::cout << "pos.see() only: " << iterations << " calls in " << elapsedUs << "us -> "
                  << (double(elapsedUs) / iterations) << "us/call\n";
    }

    return 0;
}
