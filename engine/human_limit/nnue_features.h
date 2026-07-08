#pragma once

#include <vector>

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

int featureIndexForPiece(const PerspectiveContext& ctx, int r, int c, char piece);

struct ThreatFact {
    bool attackerIsWhite;
    int attackerType;
    int victimType;
};

std::vector<ThreatFact> computeThreatFacts(const chess::BoardArray& board);

std::vector<int> threatFeaturesForPerspective(const std::vector<ThreatFact>& facts, int kingBucket,
                                               bool perspIsWhite);

std::vector<int> computeThreatFeatures(const chess::BoardArray& board, bool perspIsWhite);

}
