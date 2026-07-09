#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../engine/human_limit/nnue_features.h"

namespace {

int failures = 0;
int totalPositionsChecked = 0;

void check(bool cond, const std::string& label) {
    if (!cond) {
        std::cout << "FAIL: " << label << "\n";
        failures++;
    }
}

bool factLess(const human_limit::ThreatFact& a, const human_limit::ThreatFact& b) {
    if (a.attackerIsWhite != b.attackerIsWhite) return a.attackerIsWhite < b.attackerIsWhite;
    if (a.attackerType != b.attackerType) return a.attackerType < b.attackerType;
    return a.victimType < b.victimType;
}

void checkPosition(const chess::bitboard::Position& pos, const std::string& label) {
    auto board = pos.toBoardArray();
    auto slow = human_limit::computeThreatFacts(board);
    auto fast = human_limit::computeThreatFactsBB(pos);

    std::sort(slow.begin(), slow.end(), factLess);
    std::sort(fast.begin(), fast.end(), factLess);

    bool same = slow.size() == fast.size();
    if (same) {
        for (size_t i = 0; i < slow.size(); i++) {
            if (slow[i].attackerIsWhite != fast[i].attackerIsWhite ||
                slow[i].attackerType != fast[i].attackerType ||
                slow[i].victimType != fast[i].victimType) {
                same = false;
                break;
            }
        }
    }
    check(same, label + " threats (slow=" + std::to_string(slow.size()) + " fast=" + std::to_string(fast.size()) + ")");

    for (bool perspIsWhite : {true, false}) {
        auto slowCtx = human_limit::computePerspectiveContext(board, perspIsWhite);
        auto fastCtx = human_limit::computePerspectiveContextBB(pos, perspIsWhite);
        bool ctxSame = slowCtx.perspIsWhite == fastCtx.perspIsWhite && slowCtx.mirror == fastCtx.mirror &&
                       slowCtx.kingBucket == fastCtx.kingBucket;
        check(ctxSame, label + " perspectiveContext persp=" + std::to_string(perspIsWhite));
    }

    totalPositionsChecked++;
}

void runRandomWalk(const std::string& fen, int plies, uint32_t seed, const std::string& label) {
    chess::bitboard::Position pos(fen);
    std::mt19937 rng(seed);

    for (int ply = 0; ply < plies; ply++) {
        checkPosition(pos, label + " ply " + std::to_string(ply));
        auto moves = pos.getValidMoves();
        if (moves.empty()) break;
        std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
        auto m = moves[dist(rng)];
        char promo = (m.promotion == ' ') ? 'q' : m.promotion;
        pos.makeMove(m.from, m.to, promo);
    }
}

}

int main() {
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    for (uint32_t seed = 0; seed < 20; seed++) {
        runRandomWalk("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 60, seed,
                      "startpos walk seed=" + std::to_string(seed));
    }

    // Hand-crafted tactical positions: dense middlegame, pins on every axis,
    // double check, endgame with few pieces, promotions pending.
    std::vector<std::pair<std::string, std::string>> handCrafted = {
        {"r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "italian opening"},
        {"r1b2rk1/1pq1bppp/p1n1pn2/3p4/2PP4/1PN1PN2/PB1QBPPP/R4RK1 w - - 0 1", "dense middlegame"},
        {"1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1", "queen sac tactic"},
        {"8/3R3P/1p2kB2/2p5/p1Pp1P2/P2P1b2/7K/8 b - - 0 52", "rook-up endgame"},
        {"r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "kiwipete"},
        {"4k3/8/8/8/8/8/4P3/4K2R w K - 0 1", "sparse endgame with castling"},
        {"r6k/8/8/8/8/8/8/R3K2R w KQ - 0 1", "rook pin candidates"},
        {"8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "en passant heavy"},
        {"rnbq1bnr/pppPkppp/8/4p3/4P3/8/PPP2PPP/RNBQKBNR w KQ - 1 5", "pending promotion"},
    };
    for (auto& [fen, name] : handCrafted) {
        chess::bitboard::Position pos(fen);
        checkPosition(pos, name);
        // also walk a few plies forward from each to broaden coverage
        std::mt19937 rng(42);
        for (int ply = 0; ply < 15; ply++) {
            auto moves = pos.getValidMoves();
            if (moves.empty()) break;
            std::uniform_int_distribution<size_t> dist(0, moves.size() - 1);
            auto m = moves[dist(rng)];
            char promo = (m.promotion == ' ') ? 'q' : m.promotion;
            pos.makeMove(m.from, m.to, promo);
            checkPosition(pos, name + " +" + std::to_string(ply + 1) + "ply");
        }
    }

    std::cout << "checked " << totalPositionsChecked << " positions, " << failures << " failures\n";
    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
