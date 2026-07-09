#pragma once

#include <vector>

#include "../../chess/bitboard/position.h"
#include "../../chess/pieces.h"

namespace human_limit {

constexpr int kNumKingBuckets = 32;
constexpr int kNumPieceTypeSlots = 11;
constexpr int kSlotsPerBucket = kNumPieceTypeSlots * 64;
constexpr int kNumPlacementFeatures = kNumKingBuckets * kSlotsPerBucket;

constexpr int kNumThreatPieceTypes = 6;
constexpr int kNumThreatRelations = 2;
constexpr int kThreatsPerBucket = kNumThreatRelations * kNumThreatPieceTypes * kNumThreatPieceTypes;
constexpr int kNumThreatFeatures = kNumKingBuckets * kThreatsPerBucket;

constexpr int kNumFeatures = kNumPlacementFeatures + kNumThreatFeatures;
constexpr int kNumOutputBuckets = 8;

int computeOutputBucket(const chess::BoardArray& board);
int outputBucketFromPieceCount(int pieceCount);

struct FeatureSet {
    std::vector<int> stm;
    std::vector<int> ntm;
};

FeatureSet extractNnueFeatures(const chess::BoardArray& board, chess::Color sideToMove);

struct PerspectiveContext {
    bool perspIsWhite;
    bool mirror;
    int kingBucket;
};

PerspectiveContext computePerspectiveContext(const chess::BoardArray& board, bool perspIsWhite);

// Bitboard-native equivalent, using Position::kingSquare() (O(1)) instead of an
// O(64) mailbox scan. Must match computePerspectiveContext() exactly (see
// tests/test_threat_facts_cross_validate.cpp).
PerspectiveContext computePerspectiveContextBB(const chess::bitboard::Position& pos, bool perspIsWhite);

int featureIndexForPiece(const PerspectiveContext& ctx, int r, int c, char piece);

struct ThreatFact {
    bool attackerIsWhite;
    int attackerType;
    int victimType;
};

std::vector<ThreatFact> computeThreatFacts(const chess::BoardArray& board);

// Bitboard-native equivalent of computeThreatFacts(), using Position's existing
// attack tables (O(1) attackersTo() per piece) instead of mailbox geometric
// ray-walking. Must produce bit-for-bit identical output (see
// tests/test_threat_facts_cross_validate.cpp) -- this is the hot-path version
// used by Searcher::evalWhiteRelative; the mailbox version above remains the
// verified ground truth.
std::vector<ThreatFact> computeThreatFactsBB(const chess::bitboard::Position& pos);

std::vector<int> threatFeaturesForPerspective(const std::vector<ThreatFact>& facts, int kingBucket,
                                               bool perspIsWhite);

std::vector<int> computeThreatFeatures(const chess::BoardArray& board, bool perspIsWhite);

}
