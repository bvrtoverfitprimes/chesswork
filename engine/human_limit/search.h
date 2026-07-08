#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "../../chess/bitboard/position.h"
#include "accumulator.h"
#include "network.h"

namespace human_limit {

struct SearchResult {
    std::string uci;
    int score = 0;
    int depthReached = 0;
    long nodes = 0;
};

constexpr chess::bitboard::BBMove kNoMove{-1, -1, ' '};

class Searcher {
public:
    explicit Searcher(const Network& net);

    SearchResult findBestMove(chess::bitboard::Position& pos, int maxDepth, int timeMs);

private:
    const Network& net_;

    struct TTEntry {
        uint64_t key = 0;
        int depth = -1;
        int score = 0;
        int flag = 0;
        chess::bitboard::BBMove bestMove = kNoMove;
    };

    static constexpr size_t kTTSize = 1u << 22;
    static constexpr int kMaxPly = 128;
    static constexpr int kMaxAccDepth = 512;

    std::vector<TTEntry> tt_ = std::vector<TTEntry>(kTTSize);
    chess::bitboard::BBMove killers_[kMaxPly][2];
    int history_[64][64] = {};

    static constexpr int kNumPieceKinds = 12;
    static constexpr int kContHistDim = kNumPieceKinds * 64;
    std::vector<int> contHist_ = std::vector<int>(static_cast<size_t>(kContHistDim) * kContHistDim, 0);
    static int pieceKindIndex(char mailboxPiece);
    static int contHistIndex(char prevPiece, int prevTo, char piece, int to);
    static constexpr int kNoPieceTo = -1;
    int prevMovePieceTo_[kMaxPly] = {};

    std::vector<Accumulator> accStack_ = std::vector<Accumulator>(kMaxAccDepth);
    int accTop_ = 0;

    int staticEvalStack_[kMaxPly] = {};

    static constexpr int kCorrHistBits = 14;
    static constexpr size_t kCorrHistSize = 1u << kCorrHistBits;
    static constexpr int kCorrHistLimit = 1024;
    int corrHist_[2][kCorrHistSize] = {};

    size_t correctionKey(const chess::bitboard::Position& pos) const;
    int correctedStaticEval(chess::bitboard::Position& pos);
    void updateCorrectionHistory(chess::bitboard::Position& pos, int depth, int staticEval, int searchResult);

    std::chrono::steady_clock::time_point startTime_;
    int timeLimitMs_ = 0;
    bool stopped_ = false;
    long nodeCount_ = 0;

    bool timeExpired();
    std::vector<chess::bitboard::BBMove> orderMoves(chess::bitboard::Position& pos,
                                                      const std::vector<chess::bitboard::BBMove>& moves, int ply,
                                                      const chess::bitboard::BBMove& ttMove);
    int lmrReduction(int depth, int moveIndex) const;
    int evalWhiteRelative(chess::bitboard::Position& pos);
    void pushMove(chess::bitboard::Position& pos, const chess::bitboard::BBUndo& undo);
    void popMove();
    void pushNull();
    void popNull();
    int quiescence(chess::bitboard::Position& pos, int ply, int alpha, int beta);
    int negamax(chess::bitboard::Position& pos, int depth, int ply, int alpha, int beta, bool allowNull,
                const chess::bitboard::BBMove& excludedMove = kNoMove);
};

}
