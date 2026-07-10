#include "search.h"

#include "../raw_engine/evaluation.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

namespace human_limit {

namespace {

using chess::bitboard::BBMove;
using chess::bitboard::Position;

constexpr int kInfinity = 10'000'000;
constexpr int kMateScore = 1'000'000;
constexpr int kMaxPly = 128;

constexpr int kLmrMaxDepth = 64;
constexpr int kLmrMaxMoveIndex = 64;
int g_lmrTable[kLmrMaxDepth][kLmrMaxMoveIndex];
struct LmrTableInit {
    LmrTableInit() {
        for (int d = 1; d < kLmrMaxDepth; d++) {
            for (int mi = 1; mi < kLmrMaxMoveIndex; mi++) {
                double r = 1.0 + std::log(static_cast<double>(d)) * std::log(static_cast<double>(mi)) * 0.75;
                g_lmrTable[d][mi] = static_cast<int>(r);
            }
        }
    }
} g_lmrTableInit;

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

std::string moveToUci(const BBMove& m) {
    std::string uci = Position::squareToUci(m.from) + Position::squareToUci(m.to);
    if (m.promotion != ' ') uci += m.promotion;
    return uci;
}

bool isCaptureMove(const Position& pos, const BBMove& m) {
    if (pos.pieceAt(m.to) != ' ') return true;
    char piece = pos.pieceAt(m.from);
    return std::tolower(static_cast<unsigned char>(piece)) == 'p' && pos.enPassantTarget() == m.to;
}

int capturedPieceValue(const Position& pos, const BBMove& m) {
    char captured = pos.pieceAt(m.to);
    if (captured != ' ') return pieceValue(std::tolower(static_cast<unsigned char>(captured)));
    return isCaptureMove(pos, m) ? pieceValue('p') : 0;
}

int scoreToTT(int score, int ply) {
    if (score >= kMateScore - kMaxPly) return score + ply;
    if (score <= -(kMateScore - kMaxPly)) return score - ply;
    return score;
}

int scoreFromTT(int score, int ply) {
    if (score >= kMateScore - kMaxPly) return score - ply;
    if (score <= -(kMateScore - kMaxPly)) return score + ply;
    return score;
}

constexpr int kRfpMarginPerPly = 150;
constexpr int kRfpMaxDepth = 8;

constexpr int kFutilityMargin[4] = {0, 100, 300, 500};
constexpr int kFutilityMaxDepth = 3;

constexpr int kRazorMargin[4] = {0, 300, 500, 800};
constexpr int kRazorMaxDepth = 3;

constexpr int kLmpMaxDepth = 6;
int lmpThreshold(int depth, bool improving) {
    return improving ? (5 + depth * depth) : (3 + depth * depth / 2);
}

constexpr int kIirMinDepth = 6;
constexpr int kSingularMinDepth = 6;
constexpr int kProbCutMinDepth = 5;
constexpr int kProbCutMargin = 200;

constexpr int kDeltaMargin = 200;

}

Searcher::Searcher(const Network& net)
    : net_(net), ttShared_(std::make_shared<SharedTT>()) {
    tt_ = ttShared_->entries.data();
    for (auto& row : killers_) row[0] = row[1] = kNoMove;
    for (auto& v : prevMovePieceTo_) v = kNoPieceTo;
    if (const char* rw = std::getenv("RAW_WEIGHT")) rawWeight_ = std::atof(rw);
    if (const char* ag = std::getenv("AGREE_GATE")) agreeGate_ = std::atoi(ag);
}

Searcher::Searcher(const Network& net, std::shared_ptr<SharedTT> sharedTT)
    : net_(net), ttShared_(std::move(sharedTT)), isHelper_(true) {
    tt_ = ttShared_->entries.data();
    for (auto& row : killers_) row[0] = row[1] = kNoMove;
    for (auto& v : prevMovePieceTo_) v = kNoPieceTo;
    if (const char* rw = std::getenv("RAW_WEIGHT")) rawWeight_ = std::atof(rw);
    if (const char* ag = std::getenv("AGREE_GATE")) agreeGate_ = std::atoi(ag);
}

int Searcher::pieceKindIndex(char mailboxPiece) {
    static const std::string kOrder = "PNBRQKpnbrqk";
    size_t pos = kOrder.find(mailboxPiece);
    return pos == std::string::npos ? 0 : static_cast<int>(pos);
}

int Searcher::contHistIndex(char prevPiece, int prevTo, char piece, int to) {
    int prevIdx = pieceKindIndex(prevPiece) * 64 + prevTo;
    int idx = pieceKindIndex(piece) * 64 + to;
    return prevIdx * kContHistDim + idx;
}

int Searcher::lmrReduction(int depth, int moveIndex) const {
    int d = std::min(depth, kLmrMaxDepth - 1);
    int mi = std::min(moveIndex, kLmrMaxMoveIndex - 1);
    if (d < 1 || mi < 1) return 0;
    return g_lmrTable[d][mi];
}

size_t Searcher::correctionKey(const Position& pos) const {
    uint64_t h = pos.pawnBitboard(chess::Color::White) * 0x9E3779B97F4A7C15ULL;
    h ^= pos.pawnBitboard(chess::Color::Black) * 0xC2B2AE3D27D4EB4FULL;
    h ^= pos.nonPawnBitboard(chess::Color::White) * 0x165667B19E3779F9ULL;
    h ^= pos.nonPawnBitboard(chess::Color::Black) * 0x27D4EB2F165667C5ULL;
    return static_cast<size_t>(h >> (64 - kCorrHistBits));
}

int Searcher::correctedStaticEval(Position& pos) {
    int se = evalWhiteRelative(pos);
    int stmRelative = (pos.turn() == chess::Color::White) ? se : -se;
    int sideIdx = (pos.turn() == chess::Color::White) ? 0 : 1;
    int correction = corrHist_[sideIdx][correctionKey(pos)] / 32;
    return stmRelative + correction;
}

void Searcher::updateCorrectionHistory(Position& pos, int depth, int staticEval, int searchResult) {
    int sideIdx = (pos.turn() == chess::Color::White) ? 0 : 1;
    size_t key = correctionKey(pos);
    int bonus = std::clamp((searchResult - staticEval) * depth / 8, -kCorrHistLimit / 4, kCorrHistLimit / 4);
    int& entry = corrHist_[sideIdx][key];
    entry += bonus - entry * std::abs(bonus) / kCorrHistLimit;
    entry = std::clamp(entry, -kCorrHistLimit, kCorrHistLimit);
}

int Searcher::evalWhiteRelative(Position& pos) {
    if (rawWeight_ >= 0.999) {
        nodeDisagree_ = 0;
        uint64_t rkey = pos.zobristHash();
        size_t rslot = rkey & (kEvalCacheSize - 1);
        if (evalCacheKeys_[rslot] == rkey) return evalCacheVals_[rslot];
        int rv = raw_engine::evaluateWhiteRelative(pos);
        evalCacheKeys_[rslot] = rkey;
        evalCacheVals_[rslot] = rv;
        return rv;
    }
    uint64_t key = pos.zobristHash();
    size_t slot = key & (kEvalCacheSize - 1);
    int nnue;
    if (evalCacheKeys_[slot] == key) {
        nnue = evalCacheVals_[slot];
    } else {
        const Accumulator& acc = accStack_[accTop_];
        int bucket = outputBucketFromPieceCount(acc.pieceCount);
        nnue = static_cast<int>(net_.evaluateFromAccumulatorsWithThreats(acc.white, acc.black, pos, pos.turn(), bucket));
        evalCacheKeys_[slot] = key;
        evalCacheVals_[slot] = nnue;
    }
    if (rawWeight_ <= 0.001 && agreeGate_ <= 0) return nnue;
    int raw = raw_engine::evaluateWhiteRelative(pos);
    nodeDisagree_ = std::abs(nnue - raw);
    return static_cast<int>((1.0 - rawWeight_) * nnue + rawWeight_ * raw);
}

void Searcher::pushMove(Position& pos, const chess::bitboard::BBUndo& undo) {
    if (rawWeight_ < 0.999) {
        applyMoveToAccumulator(net_, pos, undo, accStack_[accTop_], &accStack_[accTop_ + 1]);
    }
    accTop_++;
}

void Searcher::popMove() { accTop_--; }

void Searcher::pushNull() {
    accStack_[accTop_ + 1] = accStack_[accTop_];
    accTop_++;
}

void Searcher::popNull() { accTop_--; }

bool Searcher::timeExpired() {
    if (ttShared_ && ttShared_->stop.load(std::memory_order_relaxed)) return true;
    if (timeLimitMs_ <= 0) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - startTime_)
                       .count();
    return elapsed >= timeLimitMs_;
}

std::vector<BBMove> Searcher::orderMoves(Position& pos, const std::vector<BBMove>& moves, int ply,
                                          const BBMove& ttMove) {
    int clampedPly = std::min(ply, kMaxPly - 1);
    int prevEnc = prevMovePieceTo_[clampedPly];

    auto scoreOf = [&](const BBMove& m) {
        if (m == ttMove) return 1'000'000;

        if (isCaptureMove(pos, m)) {
            int see = pos.see(m.from, m.to);
            return see >= 0 ? 100'000 + see : -100'000 + see;
        }
        if (m == killers_[clampedPly][0]) return 90'000;
        if (m == killers_[clampedPly][1]) return 89'000;

        int score = history_[m.from][m.to];
        if (prevEnc != kNoPieceTo) {
            char piece = pos.pieceAt(m.from);
            int idx = pieceKindIndex(piece) * 64 + m.to;
            score += contHist_[static_cast<size_t>(prevEnc) * kContHistDim + idx];
        }
        return score;
    };

    struct Entry { int score; int origIdx; BBMove move; };
    Entry scored[kMaxMoves];
    int n = static_cast<int>(moves.size() < kMaxMoves ? moves.size() : kMaxMoves);
    for (int i = 0; i < n; i++) scored[i] = {scoreOf(moves[i]), i, moves[i]};

    // sort with index tiebreak == stable_sort, but no heap allocation
    std::sort(scored, scored + n, [](const Entry& a, const Entry& b) {
        return a.score != b.score ? a.score > b.score : a.origIdx < b.origIdx;
    });

    std::vector<BBMove> ordered;
    ordered.reserve(n);
    for (int i = 0; i < n; i++) ordered.push_back(scored[i].move);
    return ordered;
}

int Searcher::quiescence(Position& pos, int ply, int alpha, int beta) {
    if (stopped_) return 0;
    nodeCount_++;
    if ((nodeCount_ & 2047) == 0 && timeExpired()) {
        stopped_ = true;
        return 0;
    }

    bool inCheck = pos.inCheck();
    int standPat = evalWhiteRelative(pos);
    int score = (pos.turn() == chess::Color::White) ? standPat : -standPat;

    if (!inCheck) {
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }

    auto moves = pos.getValidMoves();
    if (moves.empty()) return inCheck ? -(kMateScore - ply) : 0;

    std::vector<BBMove> candidates;
    if (inCheck) {
        candidates = moves;
    } else {
        for (const auto& m : moves) {
            if (!isCaptureMove(pos, m)) continue;
            if (pos.see(m.from, m.to) < 0) continue;
            if (score + capturedPieceValue(pos, m) + kDeltaMargin < alpha) continue;
            candidates.push_back(m);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [&](const BBMove& a, const BBMove& b) {
        if (inCheck) {
            return capturedPieceValue(pos, a) > capturedPieceValue(pos, b);
        }
        return pos.see(a.from, a.to) > pos.see(b.from, b.to);
    });

    for (const auto& m : candidates) {
        char promo = (m.promotion != ' ') ? m.promotion : 'q';
        auto undo = pos.makeMove(m.from, m.to, promo);
        pushMove(pos, undo);
        int childScore = -quiescence(pos, ply + 1, -beta, -alpha);
        popMove();
        pos.unmakeMove(undo);

        if (stopped_) return 0;
        if (childScore >= beta) return beta;
        if (childScore > alpha) alpha = childScore;
    }

    return alpha;
}

int Searcher::negamax(Position& pos, int depth, int ply, int alpha, int beta, bool allowNull,
                       const BBMove& excludedMove) {
    if (stopped_) return 0;
    nodeCount_++;
    if ((nodeCount_ & 2047) == 0 && timeExpired()) {
        stopped_ = true;
        return 0;
    }

    if (ply >= kMaxPly - 1) {
        int whiteScore = evalWhiteRelative(pos);
        return (pos.turn() == chess::Color::White) ? whiteScore : -whiteScore;
    }

    if (ply > 0 && (pos.isFiftyMoveDraw() || pos.isRepetitionDraw() || pos.isInsufficientMaterial())) {
        return 0;
    }

    if (ply > 0) {
        int mateAlpha = -(kMateScore - ply);
        int mateBeta = kMateScore - ply - 1;
        if (alpha < mateAlpha) alpha = mateAlpha;
        if (beta > mateBeta) beta = mateBeta;
        if (alpha >= beta) return alpha;
    }

    bool isExcludedSearch = !(excludedMove == kNoMove);

    uint64_t key = pos.zobristHash();
    size_t idx = key & (kTTSize - 1);
    int alphaOrig = alpha;
    BBMove ttMove = kNoMove;
    bool ttHit = (tt_[idx].key == key);
    int ttFlag = ttHit ? tt_[idx].flag : -1;
    int ttScore = ttHit ? scoreFromTT(tt_[idx].score, ply) : 0;
    int ttDepth = ttHit ? tt_[idx].depth : -1;

    if (ttHit) {
        ttMove = tt_[idx].bestMove;
        if (!isExcludedSearch && ttDepth >= depth) {
            if (ttFlag == 0) {
                return ttScore;
            } else if (ttFlag == 1) {
                alpha = std::max(alpha, ttScore);
            } else if (ttFlag == 2) {
                beta = std::min(beta, ttScore);
            }
            if (alpha >= beta) return ttScore;
        }
    }

    if (!isExcludedSearch && depth >= kIirMinDepth && ttMove == kNoMove) depth--;

    bool inCheck = pos.inCheck();

    if (depth <= 0) {
        return quiescence(pos, ply, alpha, beta);
    }

    bool nearMate = std::abs(beta) >= kMateScore - kMaxPly;
    int staticEval = 0;
    bool haveStaticEval = false;
    auto getStaticEval = [&]() {
        if (!haveStaticEval) {
            staticEval = correctedStaticEval(pos);
            haveStaticEval = true;
        }
        return staticEval;
    };

    int clampedPly = std::min(ply, kMaxPly - 1);
    bool improving = false;
    if (!inCheck) {
        int se = getStaticEval();
        staticEvalStack_[clampedPly] = se;
        improving = ply >= 2 && se > staticEvalStack_[clampedPly - 2];
    } else if (clampedPly >= 2) {
        staticEvalStack_[clampedPly] = staticEvalStack_[clampedPly - 2];
    }

    if (!inCheck && ply > 0 && depth <= kRfpMaxDepth && !nearMate) {
        int se = getStaticEval();
        int margin = improving ? 100 * depth : kRfpMarginPerPly * depth;
        if (se - margin >= beta && trustEval()) return beta;
    }

    if (!isExcludedSearch && !inCheck && ply > 0 && depth <= kRazorMaxDepth && !nearMate &&
        getStaticEval() + kRazorMargin[depth] <= alpha && trustEval()) {
        int razorScore = quiescence(pos, ply, alpha, beta);
        if (razorScore <= alpha) return razorScore;
    }

    if (allowNull && !isExcludedSearch && !inCheck && ply > 0 && depth >= 3 &&
        pos.hasNonPawnMaterial(pos.turn()) && getStaticEval() >= beta && trustEval()) {
        int r = 3 + depth / 4 + std::min((getStaticEval() - beta) / 200, 3);
        int nullDepth = depth - 1 - r;
        auto nullUndo = pos.makeNullMove();
        pushNull();
        int clampedNullPly = std::min(ply + 1, kMaxPly - 1);
        prevMovePieceTo_[clampedNullPly] = kNoPieceTo;
        int score = -negamax(pos, nullDepth, ply + 1, -beta, -beta + 1, false);
        popNull();
        pos.unmakeNullMove(nullUndo);
        if (stopped_) return 0;
        if (score >= beta) {
            if (score >= kMateScore - kMaxPly) score = beta;
            if (depth < 12) return score;
            int verify = negamax(pos, depth - 1 - r, ply, alpha, beta, false);
            if (verify >= beta) return verify;
        }
    }

    auto moves = pos.getValidMoves();
    if (moves.empty()) return inCheck ? -(kMateScore - ply) : 0;

    auto ordered = orderMoves(pos, moves, ply, ttMove);

    if (!isExcludedSearch && !inCheck && ply > 0 && depth >= kProbCutMinDepth && !nearMate) {
        int probCutBeta = beta + kProbCutMargin;
        int seThreshold = probCutBeta - getStaticEval();
        for (const auto& m : ordered) {
            if (!isCaptureMove(pos, m)) continue;
            if (pos.see(m.from, m.to) < seThreshold) continue;
            char promo = (m.promotion != ' ') ? m.promotion : 'q';
            auto undo = pos.makeMove(m.from, m.to, promo);
            pushMove(pos, undo);
            int score = -negamax(pos, depth - 4, ply + 1, -probCutBeta, -probCutBeta + 1, true);
            popMove();
            pos.unmakeMove(undo);
            if (stopped_) return 0;
            if (score >= probCutBeta) return score;
        }
    }

    int best = -kInfinity;
    BBMove bestMove = kNoMove;
    int moveIndex = 0;
    bool anyMoveSearched = false;
    int prevEnc = prevMovePieceTo_[clampedPly];
    std::vector<BBMove> triedQuiets;
    triedQuiets.reserve(8);

    for (const auto& m : ordered) {
        if (m == excludedMove) continue;

        char promo = (m.promotion != ' ') ? m.promotion : 'q';
        bool isCapture = isCaptureMove(pos, m);
        bool isPromotion = m.promotion != ' ';
        bool quiet = !isCapture && !isPromotion;
        bool losingCapture = isCapture && !isPromotion && pos.see(m.from, m.to) < 0;

        int extension = 0;
        if (!isExcludedSearch && ply > 0 && m == ttMove && depth >= kSingularMinDepth && ttHit &&
            ttFlag == 1 && ttDepth >= depth - 3 && std::abs(ttScore) < kMateScore - kMaxPly) {
            int singularBeta = ttScore - depth;
            int singularDepth = (depth - 1) / 2;
            int excludedScore = negamax(pos, singularDepth, ply, singularBeta - 1, singularBeta, true, ttMove);
            if (stopped_) return 0;
            if (excludedScore < singularBeta) {
                extension = 1;
            } else if (singularBeta >= beta) {
                return singularBeta;
            }
        }

        auto undo = pos.makeMove(m.from, m.to, promo);
        __builtin_prefetch(&tt_[pos.zobristHash() & (kTTSize - 1)]);
        pushMove(pos, undo);
        bool givesCheck = pos.inCheck();
        if (givesCheck && !losingCapture) extension = std::max(extension, 1);
        int childPly = std::min(ply + 1, kMaxPly - 1);
        prevMovePieceTo_[childPly] = pieceKindIndex(undo.piece) * 64 + m.to;

        if (quiet && !inCheck && !givesCheck && ply > 0 && anyMoveSearched && !nearMate) {
            if (depth <= kLmpMaxDepth && moveIndex >= lmpThreshold(depth, improving)) {
                popMove();
                pos.unmakeMove(undo);
                moveIndex++;
                continue;
            }
            if (depth <= kFutilityMaxDepth && getStaticEval() + kFutilityMargin[depth] <= alpha && trustEval()) {
                popMove();
                pos.unmakeMove(undo);
                moveIndex++;
                continue;
            }
        }

        if (quiet && !givesCheck) triedQuiets.push_back(m);

        int score;
        if (moveIndex >= 2 && depth >= 3 && (quiet || losingCapture) && !givesCheck) {
            int r = lmrReduction(depth, moveIndex);
            if (!improving) r++;
            if (losingCapture) r++;
            int reducedDepth = std::max(1, depth - 1 - r);
            score = -negamax(pos, reducedDepth, ply + 1, -alpha - 1, -alpha, true);
            if (score > alpha && !stopped_) {
                score = -negamax(pos, depth - 1 + extension, ply + 1, -beta, -alpha, true);
            }
        } else {
            score = -negamax(pos, depth - 1 + extension, ply + 1, -beta, -alpha, true);
        }

        popMove();
        pos.unmakeMove(undo);
        if (stopped_) return 0;
        anyMoveSearched = true;

        if (score > best) {
            best = score;
            bestMove = m;
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) {
            if (!isCapture) {
                if (!(killers_[clampedPly][0] == m)) {
                    killers_[clampedPly][1] = killers_[clampedPly][0];
                    killers_[clampedPly][0] = m;
                }
                int bonus = depth * depth;
                history_[m.from][m.to] += bonus - history_[m.from][m.to] * std::abs(bonus) / 16384;
                if (prevEnc != kNoPieceTo) {
                    int idx = pieceKindIndex(undo.piece) * 64 + m.to;
                    int& e = contHist_[static_cast<size_t>(prevEnc) * kContHistDim + idx];
                    e += bonus - e * std::abs(bonus) / 16384;
                }
                for (const auto& fm : triedQuiets) {
                    if (fm == m) continue;
                    int malus = -bonus;
                    history_[fm.from][fm.to] += malus - history_[fm.from][fm.to] * std::abs(malus) / 16384;
                    if (prevEnc != kNoPieceTo) {
                        int fidx = pieceKindIndex(pos.pieceAt(fm.from)) * 64 + fm.to;
                        int& fe = contHist_[static_cast<size_t>(prevEnc) * kContHistDim + fidx];
                        fe += malus - fe * std::abs(malus) / 16384;
                    }
                }
            }
            break;
        }
        moveIndex++;
    }

    if (isExcludedSearch) return best;

    if (haveStaticEval && !inCheck && std::abs(best) < kMateScore - kMaxPly) {
        updateCorrectionHistory(pos, depth, staticEval, best);
    }

    int flag = (best <= alphaOrig) ? 2 : (best >= beta ? 1 : 0);
    tt_[idx] = {key, depth, scoreToTT(best, ply), flag, bestMove};

    return best;
}

SearchResult Searcher::findBestMove(Position& pos, int maxDepth, int timeMs) {
    if (numThreads_ <= 1) {
        return searchInternal(pos, maxDepth, timeMs, true);
    }

    // Lazy SMP: helper threads share this searcher's transposition table and
    // search their own copy of the position, diverging via TT races to help the
    // main thread reach greater depth. Only the main thread's result is used.
    ttShared_->stop.store(false, std::memory_order_relaxed);
    std::vector<std::thread> helpers;
    std::vector<std::unique_ptr<Searcher>> workers;
    std::vector<std::unique_ptr<Position>> posCopies;
    // Snapshot each helper's own position copy here in the main thread, before
    // any search starts mutating `pos`, so there is no race on the shared object.
    // Spawning is guarded: if the machine is under resource pressure and a thread
    // (or its allocations) can't be created, we degrade gracefully to fewer
    // helpers / single-thread rather than taking down the whole process.
    try {
        for (int t = 1; t < numThreads_; t++) {
            auto worker = std::make_unique<Searcher>(net_, ttShared_);
            worker->setHelperId(t);
            auto pc = std::make_unique<Position>(pos);
            Searcher* w = worker.get();
            Position* pcp = pc.get();
            // Own the objects before spawning: if thread creation throws, they
            // stay alive (owned by the vectors); the running helpers always
            // reference heap objects that outlive them.
            workers.push_back(std::move(worker));
            posCopies.push_back(std::move(pc));
            helpers.emplace_back([w, pcp, maxDepth, timeMs]() {
                try {
                    w->searchInternal(*pcp, maxDepth, timeMs, false);
                } catch (...) {
                    // A helper failing must never affect the main result.
                }
            });
        }
    } catch (...) {
        // Could not spawn all helpers; continue with however many we have.
    }

    SearchResult result = searchInternal(pos, maxDepth, timeMs, true);
    ttShared_->stop.store(true, std::memory_order_relaxed);
    for (auto& h : helpers) {
        if (h.joinable()) h.join();
    }
    ttShared_->stop.store(false, std::memory_order_relaxed);
    return result;
}

SearchResult Searcher::searchInternal(Position& pos, int maxDepth, int timeMs, bool reportVerbose) {
    startTime_ = std::chrono::steady_clock::now();
    timeLimitMs_ = timeMs;
    stopped_ = false;
    nodeCount_ = 0;

    SearchResult result;
    int prevScore = 0;

    accTop_ = 0;
    if (rawWeight_ < 0.999) initAccumulator(net_, pos, &accStack_[0]);

    auto rootMoves = pos.getValidMoves();
    if (rootMoves.empty()) return result;
    BBMove bestMoveOverall = rootMoves.front();
    result.uci = moveToUci(bestMoveOverall);

    // Lazy SMP thread diversity: helper threads skip a per-thread pattern of
    // depths so they race ahead to deeper iterations and seed the shared TT for
    // the main thread, instead of all threads duplicating the same search.
    static const int kSkipSize[20] = {1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
    static const int kSkipPhase[20] = {0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7};

    for (int depth = 1; depth <= maxDepth; depth++) {
        if (isHelper_) {
            int i = (helperId_ - 1) % 20;
            if (((depth + kSkipPhase[i]) / kSkipSize[i]) % 2 != 0) continue;
        }
        bool useAsp = !isHelper_ && depth >= 4;
        int aspirationDelta = 50;
        int windowAlpha = useAsp ? prevScore - aspirationDelta : -kInfinity;
        int windowBeta = useAsp ? prevScore + aspirationDelta : kInfinity;

        BBMove bestMoveThisDepth = kNoMove;
        int bestScoreThisDepth = -kInfinity;
        bool aborted = false;
        int aspFails = 0;

        while (true) {
            bestMoveThisDepth = kNoMove;
            bestScoreThisDepth = -kInfinity;
            int a = windowAlpha;
            int b = windowBeta;

            auto ordered = orderMoves(pos, rootMoves, 0, bestMoveOverall);
            for (const auto& m : ordered) {
                char promo = (m.promotion != ' ') ? m.promotion : 'q';
                auto undo = pos.makeMove(m.from, m.to, promo);
                pushMove(pos, undo);
                int score = -negamax(pos, depth - 1, 1, -b, -a, true);
                popMove();
                pos.unmakeMove(undo);

                if (stopped_) {
                    aborted = true;
                    break;
                }
                if (score > bestScoreThisDepth) {
                    bestScoreThisDepth = score;
                    bestMoveThisDepth = m;
                }
                if (score > a) a = score;
            }

            if (aborted) break;
            if (bestScoreThisDepth <= windowAlpha || bestScoreThisDepth >= windowBeta) {
                aspFails++;
                aspirationDelta *= 4;
                if (aspirationDelta >= kInfinity / 2) {
                    windowAlpha = -kInfinity;
                    windowBeta = kInfinity;
                } else {
                    windowAlpha = prevScore - aspirationDelta;
                    windowBeta = prevScore + aspirationDelta;
                }
                continue;
            }
            break;
        }

        if (aborted) break;

        bestMoveOverall = bestMoveThisDepth;
        result.uci = moveToUci(bestMoveOverall);
        result.score = bestScoreThisDepth;
        result.depthReached = depth;
        prevScore = bestScoreThisDepth;

        static const bool verboseEnv = std::getenv("HL_VERBOSE") != nullptr;
        if (verboseEnv && reportVerbose) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - startTime_).count();
            std::fprintf(stderr, "depth %d: %lldms nodes=%ld score=%d aspFails=%d\n", depth,
                         static_cast<long long>(ms), nodeCount_, bestScoreThisDepth, aspFails);
        }

        if (timeExpired()) break;
    }

    result.nodes = nodeCount_;
    return result;
}

}
