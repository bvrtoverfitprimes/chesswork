#include "network.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <immintrin.h>

namespace limit {

namespace {
constexpr int32_t kMagic = 0x4E4E5546;

float relu(float v) { return v > 0.0f ? v : 0.0f; }
}

namespace {

int32_t hsum256(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum4 = _mm_add_epi32(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(sum4, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sum2 = _mm_add_epi32(sum4, shuf);
    shuf = _mm_shuffle_epi32(sum2, _MM_SHUFFLE(0, 1, 0, 1));
    __m128i sum1 = _mm_add_epi32(sum2, shuf);
    return _mm_cvtsi128_si32(sum1);
}

int32_t dotU8S8Avx2(const uint8_t* u, const int8_t* s, int dim) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    int j = 0;
    for (; j + 32 <= dim; j += 32) {
        __m256i uVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(u + j));
        __m256i sVec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + j));
        __m256i prod16 = _mm256_maddubs_epi16(uVec, sVec);
        __m256i prod32 = _mm256_madd_epi16(prod16, ones16);
        acc = _mm256_add_epi32(acc, prod32);
    }
    int32_t total = hsum256(acc);
    for (; j < dim; j++) total += static_cast<int32_t>(u[j]) * static_cast<int32_t>(s[j]);
    return total;
}

}

float Network::headForward(const float* x, int bucket) const {
    const int8_t* fc1wq = &fc1wQuant_[static_cast<size_t>(bucket) * kHeadWidth * 2 * hidden_];
    const int32_t* fc1wRowSum = &fc1wRowSum_[static_cast<size_t>(bucket) * kHeadWidth];
    const float fc1Scale = fc1wScale_[bucket];
    const float* fc1b = &fc1b_[static_cast<size_t>(bucket) * kHeadWidth];
    const float* fc2w = &fc2w_[static_cast<size_t>(bucket) * kHeadWidth * kHeadWidth];
    const float* fc2b = &fc2b_[static_cast<size_t>(bucket) * kHeadWidth];
    const float* fc3w = &fc3w_[static_cast<size_t>(bucket) * kHeadWidth];
    const float fc3b = fc3b_[bucket];

    const int dim = 2 * hidden_;
    float xMaxAbs = 1e-6f;
    for (int j = 0; j < dim; j++) xMaxAbs = std::max(xMaxAbs, std::abs(x[j]));
    const float xScale = 63.0f / xMaxAbs;
    std::array<uint8_t, 2 * kMaxHidden> uxq;
    for (int j = 0; j < dim; j++) {
        int q = static_cast<int>(std::lround(x[j] * xScale));
        q = std::clamp(q, -63, 63);
        uxq[j] = static_cast<uint8_t>(q + 63);
    }
    const float descale = 1.0f / (xScale * fc1Scale);

    std::array<float, kHeadWidth> h1{};
    for (int i = 0; i < kHeadWidth; i++) {
        const int8_t* row = &fc1wq[static_cast<size_t>(i) * dim];
        int32_t rawDot = dotU8S8Avx2(uxq.data(), row, dim);
        int32_t acc = rawDot - 63 * fc1wRowSum[i];
        float s = fc1b[i] + static_cast<float>(acc) * descale;
        h1[i] = relu(s);
    }

    std::array<float, kHeadWidth> h2{};
    for (int i = 0; i < kHeadWidth; i++) {
        float s = fc2b[i];
        const float* row = &fc2w[static_cast<size_t>(i) * kHeadWidth];
        for (int j = 0; j < kHeadWidth; j++) s += row[j] * h1[j];
        h2[i] = relu(s);
    }

    float out = fc3b;
    for (int j = 0; j < kHeadWidth; j++) out += fc3w[j] * h2[j];
    return out;
}

float Network::headForwardFloatReference(const std::vector<float>& x, int bucket) const {
    const float* fc1w = &fc1w_[static_cast<size_t>(bucket) * kHeadWidth * 2 * hidden_];
    const float* fc1b = &fc1b_[static_cast<size_t>(bucket) * kHeadWidth];
    const float* fc2w = &fc2w_[static_cast<size_t>(bucket) * kHeadWidth * kHeadWidth];
    const float* fc2b = &fc2b_[static_cast<size_t>(bucket) * kHeadWidth];
    const float* fc3w = &fc3w_[static_cast<size_t>(bucket) * kHeadWidth];
    const float fc3b = fc3b_[bucket];

    std::array<float, kHeadWidth> h1{};
    for (int i = 0; i < kHeadWidth; i++) {
        float s = fc1b[i];
        const float* row = &fc1w[static_cast<size_t>(i) * 2 * hidden_];
        for (int j = 0; j < 2 * hidden_; j++) s += row[j] * x[j];
        h1[i] = relu(s);
    }

    std::array<float, kHeadWidth> h2{};
    for (int i = 0; i < kHeadWidth; i++) {
        float s = fc2b[i];
        const float* row = &fc2w[static_cast<size_t>(i) * kHeadWidth];
        for (int j = 0; j < kHeadWidth; j++) s += row[j] * h1[j];
        h2[i] = relu(s);
    }

    float out = fc3b;
    for (int j = 0; j < kHeadWidth; j++) out += fc3w[j] * h2[j];
    return out;
}

float Network::forward(const FeatureSet& f, int bucket) const {
    std::vector<float> stmAcc(hidden_, 0.0f);
    std::vector<float> ntmAcc(hidden_, 0.0f);

    for (int idx : f.stm) {
        const float* row = &embedding_[static_cast<size_t>(idx) * hidden_];
        for (int h = 0; h < hidden_; h++) stmAcc[h] += row[h];
    }
    for (int idx : f.ntm) {
        const float* row = &embedding_[static_cast<size_t>(idx) * hidden_];
        for (int h = 0; h < hidden_; h++) ntmAcc[h] += row[h];
    }

    return forwardFromAccumulators(stmAcc.data(), ntmAcc.data(), bucket);
}

float Network::forwardFromAccumulators(const float* stmAcc, const float* ntmAcc, int bucket) const {
    std::array<float, 2 * kMaxHidden> xBuf;
    for (int h = 0; h < hidden_; h++) {
        xBuf[h] = stmAcc[h];
        xBuf[hidden_ + h] = ntmAcc[h];
    }
    return headForward(xBuf.data(), bucket);
}

double Network::evaluate(const chess::Game& game) const {
    const auto& board = game.boardArray();
    FeatureSet features = extractNnueFeatures(board, game.turn());
    int bucket = computeOutputBucket(board);
    double raw = forward(features, bucket);
    double whiteRelative = (game.turn() == chess::Color::White) ? raw : -raw;
    return whiteRelative * kCpScale;
}

double Network::evaluateFromAccumulators(const std::vector<float>& whiteAcc, const std::vector<float>& blackAcc,
                                          chess::Color turn, int bucket) const {
    const float* stmAcc = (turn == chess::Color::White) ? whiteAcc.data() : blackAcc.data();
    const float* ntmAcc = (turn == chess::Color::White) ? blackAcc.data() : whiteAcc.data();
    double raw = forwardFromAccumulators(stmAcc, ntmAcc, bucket);
    double whiteRelative = (turn == chess::Color::White) ? raw : -raw;
    return whiteRelative * kCpScale;
}

double Network::evaluateFromAccumulatorsWithThreats(const std::vector<float>& whiteAcc,
                                                      const std::vector<float>& blackAcc,
                                                      const chess::bitboard::Position& pos, chess::Color turn,
                                                      int bucket) const {
    std::array<float, kMaxHidden> whiteWithThreats;
    std::array<float, kMaxHidden> blackWithThreats;
    std::copy(whiteAcc.begin(), whiteAcc.end(), whiteWithThreats.begin());
    std::copy(blackAcc.begin(), blackAcc.end(), blackWithThreats.begin());

    PerspectiveContext whiteCtx = computePerspectiveContextBB(pos, true);
    PerspectiveContext blackCtx = computePerspectiveContextBB(pos, false);
    std::vector<ThreatFact> facts = computeThreatFactsBB(pos);

    for (int idx : threatFeaturesForPerspective(facts, whiteCtx.kingBucket, true)) {
        const float* row = &embedding_[static_cast<size_t>(idx) * hidden_];
        for (int h = 0; h < hidden_; h++) whiteWithThreats[h] += row[h];
    }
    for (int idx : threatFeaturesForPerspective(facts, blackCtx.kingBucket, false)) {
        const float* row = &embedding_[static_cast<size_t>(idx) * hidden_];
        for (int h = 0; h < hidden_; h++) blackWithThreats[h] += row[h];
    }

    const float* stmAcc = (turn == chess::Color::White) ? whiteWithThreats.data() : blackWithThreats.data();
    const float* ntmAcc = (turn == chess::Color::White) ? blackWithThreats.data() : whiteWithThreats.data();
    double raw = forwardFromAccumulators(stmAcc, ntmAcc, bucket);
    double whiteRelative = (turn == chess::Color::White) ? raw : -raw;
    return whiteRelative * kCpScale;
}

bool Network::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    int32_t magic = 0, hidden = 0, numBuckets = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&hidden), sizeof(hidden));
    in.read(reinterpret_cast<char*>(&numBuckets), sizeof(numBuckets));
    if (!in || magic != kMagic || hidden <= 0 || numBuckets <= 0) return false;

    hidden_ = hidden;
    numBuckets_ = numBuckets;
    embedding_.resize(static_cast<size_t>(kNumFeatures) * hidden_);

    fc1w_.resize(static_cast<size_t>(numBuckets_) * kHeadWidth * 2 * hidden_);
    fc1b_.resize(static_cast<size_t>(numBuckets_) * kHeadWidth);
    fc2w_.resize(static_cast<size_t>(numBuckets_) * kHeadWidth * kHeadWidth);
    fc2b_.resize(static_cast<size_t>(numBuckets_) * kHeadWidth);
    fc3w_.resize(static_cast<size_t>(numBuckets_) * kHeadWidth);
    fc3b_.resize(numBuckets_);

    auto readBlock = [&](std::vector<float>& v, size_t offset, size_t count) {
        in.read(reinterpret_cast<char*>(v.data() + offset), static_cast<std::streamsize>(count * sizeof(float)));
    };

    readBlock(embedding_, 0, embedding_.size());
    if (!in.good()) return false;

    for (int b = 0; b < numBuckets_; b++) {
        readBlock(fc1w_, static_cast<size_t>(b) * kHeadWidth * 2 * hidden_, static_cast<size_t>(kHeadWidth) * 2 * hidden_);
        readBlock(fc1b_, static_cast<size_t>(b) * kHeadWidth, kHeadWidth);
        readBlock(fc2w_, static_cast<size_t>(b) * kHeadWidth * kHeadWidth, static_cast<size_t>(kHeadWidth) * kHeadWidth);
        readBlock(fc2b_, static_cast<size_t>(b) * kHeadWidth, kHeadWidth);
        readBlock(fc3w_, static_cast<size_t>(b) * kHeadWidth, kHeadWidth);
        readBlock(fc3b_, b, 1);
        if (!in.good()) return false;
    }

    fc1wQuant_.resize(fc1w_.size());
    fc1wScale_.resize(numBuckets_);
    fc1wRowSum_.resize(static_cast<size_t>(numBuckets_) * kHeadWidth);
    const size_t fc1RowsPerBucket = static_cast<size_t>(kHeadWidth) * 2 * hidden_;
    for (int b = 0; b < numBuckets_; b++) {
        size_t base = static_cast<size_t>(b) * fc1RowsPerBucket;
        float maxAbs = 1e-6f;
        for (size_t k = 0; k < fc1RowsPerBucket; k++) maxAbs = std::max(maxAbs, std::abs(fc1w_[base + k]));
        float scale = 127.0f / maxAbs;
        fc1wScale_[b] = scale;
        for (size_t k = 0; k < fc1RowsPerBucket; k++) {
            int q = static_cast<int>(std::lround(fc1w_[base + k] * scale));
            fc1wQuant_[base + k] = static_cast<int8_t>(std::clamp(q, -127, 127));
        }
        for (int i = 0; i < kHeadWidth; i++) {
            int32_t rowSum = 0;
            const int8_t* row = &fc1wQuant_[base + static_cast<size_t>(i) * 2 * hidden_];
            for (int j = 0; j < 2 * hidden_; j++) rowSum += row[j];
            fc1wRowSum_[static_cast<size_t>(b) * kHeadWidth + i] = rowSum;
        }
    }

    return true;
}

}
