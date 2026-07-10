#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../chess/board.h"
#include "../engine/limit/nnue_features.h"

namespace {

int failures = 0;

void check(bool cond, const std::string& label) {
    if (!cond) {
        std::cout << "FAIL: " << label << "\n";
        failures++;
    } else {
        std::cout << "PASS: " << label << "\n";
    }
}

std::vector<int> parseIntList(const std::string& csv) {
    std::vector<int> out;
    if (csv.empty()) return out;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stoi(item));
    return out;
}

}

int main() {
    std::ifstream in("training/encoding/golden_features.txt");
    check(static_cast<bool>(in), "golden fixture file opens");
    if (!in) {
        std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
        return failures == 0 ? 0 : 1;
    }

    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        lineNo++;
        size_t p1 = line.find('|');
        size_t p2 = line.find('|', p1 + 1);
        size_t p3 = line.find('|', p2 + 1);
        std::string fen = line.substr(0, p1);
        std::vector<int> expectedStm = parseIntList(line.substr(p1 + 1, p2 - p1 - 1));
        std::vector<int> expectedNtm = parseIntList(line.substr(p2 + 1, p3 - p2 - 1));
        int expectedBucket = std::stoi(line.substr(p3 + 1));

        chess::Game game(fen);
        auto features = limit::extractNnueFeatures(game.boardArray(), game.turn());
        int actualBucket = limit::computeOutputBucket(game.boardArray());

        std::vector<int> actualStm = features.stm;
        std::vector<int> actualNtm = features.ntm;
        std::sort(actualStm.begin(), actualStm.end());
        std::sort(actualNtm.begin(), actualNtm.end());

        check(actualStm == expectedStm, "golden fixture " + std::to_string(lineNo) + " stm: " + fen);
        check(actualNtm == expectedNtm, "golden fixture " + std::to_string(lineNo) + " ntm: " + fen);
        check(actualBucket == expectedBucket, "golden fixture " + std::to_string(lineNo) + " bucket: " + fen);
    }

    check(lineNo > 0, "at least one golden fixture was read");

    std::cout << (failures == 0 ? "ALL TESTS PASSED" : "TESTS FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
