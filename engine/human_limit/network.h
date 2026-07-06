#pragma once

#include <array>
#include <string>

#include "../../chess/board.h"

namespace human_limit {

constexpr int kInputSize = 20;
constexpr int kHidden1Size = 128;
constexpr int kHidden2Size = 64;
constexpr int kHidden3Size = 32;

struct Features {
    std::array<double, kInputSize> x;
    double classicalEval = 0.0;
};

constexpr double kClassicalWeight = 0.3;
constexpr double kNetworkWeight = 0.7;

Features extractFeatures(const chess::Game& game);

class Network {
public:
    Network();

    double evaluate(const chess::Game& game) const;

    double forward(const Features& f, std::array<double, kHidden1Size>* h1Out = nullptr,
                    std::array<double, kHidden2Size>* h2Out = nullptr,
                    std::array<double, kHidden3Size>* h3Out = nullptr) const;

    void trainStep(const Features& f, double target, double learningRate);

    void save(const std::string& path) const;
    bool load(const std::string& path);

private:
    std::array<std::array<double, kInputSize>, kHidden1Size> w1_;
    std::array<double, kHidden1Size> b1_;
    std::array<std::array<double, kHidden1Size>, kHidden2Size> w2_;
    std::array<double, kHidden2Size> b2_;
    std::array<std::array<double, kHidden2Size>, kHidden3Size> w3_;
    std::array<double, kHidden3Size> b3_;
    std::array<double, kHidden3Size> w4_;
    double b4_ = 0.0;

    void randomInit();
};

}
