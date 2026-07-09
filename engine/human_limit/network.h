#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../../chess/board.h"
#include "nnue_features.h"

namespace human_limit {

constexpr double kCpScale = 400.0;
constexpr int kHeadWidth = 32;

class Network {
public:
    Network() = default;

    double evaluate(const chess::Game& game) const;
    float forward(const FeatureSet& features, int bucket) const;
    float forwardFromAccumulators(const std::vector<float>& stmAcc, const std::vector<float>& ntmAcc,
                                   int bucket) const;
    double evaluateFromAccumulators(const std::vector<float>& whiteAcc, const std::vector<float>& blackAcc,
                                     chess::Color turn, int bucket) const;

    double evaluateFromAccumulatorsWithThreats(const std::vector<float>& whiteAcc, const std::vector<float>& blackAcc,
                                                const chess::BoardArray& board, chess::Color turn, int bucket) const;

    int hiddenSize() const { return hidden_; }
    const float* embeddingRow(int featureIdx) const {
        return &embedding_[static_cast<size_t>(featureIdx) * hidden_];
    }

    bool load(const std::string& path);

    float headForwardFloatReference(const std::vector<float>& x, int bucket) const;

private:
    int hidden_ = 0;
    int numBuckets_ = 0;
    std::vector<float> embedding_;

    std::vector<float> fc1w_;
    std::vector<float> fc1b_;
    std::vector<int8_t> fc1wQuant_;
    std::vector<float> fc1wScale_;
    std::vector<int32_t> fc1wRowSum_;
    std::vector<float> fc2w_;
    std::vector<float> fc2b_;
    std::vector<float> fc3w_;
    std::vector<float> fc3b_;

    float headForward(const std::vector<float>& x, int bucket) const;
};

}
