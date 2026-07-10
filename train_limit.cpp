#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

#include "chess/board.h"
#include "engine/limit/network.h"
#include "engine/limit/search.h"

namespace {

struct Sample {
    limit::Features features;
};

int pieceMaterialValue(char pieceLower) {
    switch (pieceLower) {
        case 'p': return 100;
        case 'n': return 320;
        case 'b': return 330;
        case 'r': return 500;
        case 'q': return 900;
        default: return 0;
    }
}

std::vector<std::array<char, 8>> parseFenPlacement(const std::string& placement) {
    std::vector<std::array<char, 8>> board(8);
    for (auto& row : board) row.fill(' ');
    int r = 0, c = 0;
    for (char ch : placement) {
        if (ch == '/') {
            r++;
            c = 0;
        } else if (std::isdigit(static_cast<unsigned char>(ch))) {
            c += ch - '0';
        } else {
            board[r][c] = ch;
            c++;
        }
    }
    return board;
}

std::string serializeFenPlacement(const std::vector<std::array<char, 8>>& board) {
    std::ostringstream oss;
    for (int r = 0; r < 8; r++) {
        int empty = 0;
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') {
                empty++;
            } else {
                if (empty > 0) { oss << empty; empty = 0; }
                oss << p;
            }
        }
        if (empty > 0) oss << empty;
        if (r != 7) oss << '/';
    }
    return oss.str();
}

bool generateSyntheticMaterialPosition(std::mt19937& rng, limit::Features* featuresOut, double* targetOut) {
    chess::Game game;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    int randomPlies = 4 + static_cast<int>(uni(rng) * 12);
    for (int i = 0; i < randomPlies; i++) {
        auto legal = game.getValidMovesUci();
        if (legal.empty()) return false;
        std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
        std::string chosen = legal[pick(rng)];
        chess::Pos from = chess::Game::parseSquare(chosen.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(chosen.substr(2, 2));
        char promo = (chosen.size() == 5) ? chosen[4] : 'q';
        game.makeMove(from, to, promo);
    }

    std::string fen = game.toFen();
    std::istringstream iss(fen);
    std::string placement, activeColor, castling, ep;
    int halfmove = 0;
    iss >> placement >> activeColor >> castling >> ep >> halfmove;

    auto board = parseFenPlacement(placement);

    bool weakenWhite = uni(rng) < 0.5;
    char targetCase = weakenWhite ? 'A' : 'a';
    std::vector<chess::Pos> candidates;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;
            char pl = std::tolower(static_cast<unsigned char>(p));
            if (pl == 'k' || pl == 'p') continue;
            bool isWhitePiece = std::isupper(static_cast<unsigned char>(p));
            if ((targetCase == 'A') == isWhitePiece) candidates.push_back({r, c});
        }
    }
    if (candidates.empty()) return false;

    std::shuffle(candidates.begin(), candidates.end(), rng);
    int removeCount = std::min<int>(candidates.size(), 1 + static_cast<int>(uni(rng) * 3));
    for (int i = 0; i < removeCount; i++) {
        board[candidates[i].r][candidates[i].c] = ' ';
    }

    std::string newPlacement = serializeFenPlacement(board);
    std::ostringstream newFen;
    newFen << newPlacement << " " << activeColor << " - - 0 1";

    chess::Game synthetic(newFen.str());

    int whiteMaterial = 0, blackMaterial = 0;
    const auto& finalBoard = synthetic.boardArray();
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = finalBoard[r][c];
            if (p == ' ') continue;
            int v = pieceMaterialValue(std::tolower(static_cast<unsigned char>(p)));
            if (std::isupper(static_cast<unsigned char>(p))) whiteMaterial += v;
            else blackMaterial += v;
        }
    }
    if (whiteMaterial == blackMaterial) return false;

    *featuresOut = limit::extractFeatures(synthetic);
    *targetOut = (whiteMaterial > blackMaterial) ? 1.0 : -1.0;
    return true;
}

double playSelfPlayGame(limit::Network& net, int searchDepth, int searchTimeMs, int maxPlies,
                         std::mt19937& rng, std::vector<Sample>* samplesOut) {
    chess::Game game;
    limit::Searcher searcher(net);
    std::vector<Sample> positions;

    std::uniform_real_distribution<double> uni(0.0, 1.0);

    for (int ply = 0; ply < maxPlies; ply++) {
        if (game.isFiftyMoveDraw() || game.isRepetitionDraw() || game.isInsufficientMaterial()) {
            for (auto& s : positions) samplesOut->push_back(s);
            return 0.0;
        }

        auto legal = game.getValidMovesUci();
        if (legal.empty()) {
            double result;
            if (game.inCheck()) {
                result = (game.turn() == chess::Color::White) ? -1.0 : 1.0;
            } else {
                result = 0.0;
            }
            for (auto& s : positions) samplesOut->push_back(s);
            return result;
        }

        if (ply % 3 == 0) {
            positions.push_back({limit::extractFeatures(game)});
        }

        std::string chosen;
        bool randomMove = (ply < 6) || (uni(rng) < 0.08);
        if (randomMove) {
            std::uniform_int_distribution<size_t> pick(0, legal.size() - 1);
            chosen = legal[pick(rng)];
        } else {
            auto result = searcher.findBestMove(game, searchDepth, searchTimeMs);
            chosen = result.uci;
        }

        chess::Pos from = chess::Game::parseSquare(chosen.substr(0, 2));
        chess::Pos to = chess::Game::parseSquare(chosen.substr(2, 2));
        char promo = (chosen.size() == 5) ? chosen[4] : 'q';
        game.makeMove(from, to, promo);
    }

    for (auto& s : positions) samplesOut->push_back(s);
    return 0.0;
}

}

int main(int argc, char** argv) {
    int generations = argc > 1 ? std::atoi(argv[1]) : 10;
    int gamesPerGen = argc > 2 ? std::atoi(argv[2]) : 10;
    int searchDepth = argc > 3 ? std::atoi(argv[3]) : 3;
    int searchTimeMs = argc > 4 ? std::atoi(argv[4]) : 50;
    int epochs = argc > 5 ? std::atoi(argv[5]) : 5;
    double learningRate = argc > 6 ? std::atof(argv[6]) : 0.01;
    int maxPlies = argc > 7 ? std::atoi(argv[7]) : 150;
    size_t replayCap = argc > 8 ? static_cast<size_t>(std::atoi(argv[8])) : 20000;
    int syntheticPerGen = argc > 9 ? std::atoi(argv[9]) : 20;

    const std::string weightsPath = "engine/limit/weights.txt";

    limit::Network net;
    if (net.load(weightsPath)) {
        std::cout << "Loaded existing weights from " << weightsPath << "\n";
    } else {
        std::cout << "No existing weights found, starting from random initialization\n";
    }

    std::mt19937 rng(std::random_device{}());
    std::vector<std::pair<limit::Features, double>> replayBuffer;

    for (int gen = 1; gen <= generations; gen++) {
        auto genStart = std::chrono::steady_clock::now();

        int whiteWins = 0, blackWins = 0, draws = 0;
        size_t newSamples = 0;

        for (int g = 0; g < gamesPerGen; g++) {
            std::vector<Sample> samples;
            double result = playSelfPlayGame(net, searchDepth, searchTimeMs, maxPlies, rng, &samples);
            if (result > 0.5) whiteWins++;
            else if (result < -0.5) blackWins++;
            else draws++;

            double target = result;
            for (auto& s : samples) {
                replayBuffer.push_back({s.features, target});
                newSamples++;
            }
        }

        size_t syntheticGenerated = 0;
        for (int s = 0; s < syntheticPerGen; s++) {
            limit::Features f;
            double target;
            if (generateSyntheticMaterialPosition(rng, &f, &target)) {
                replayBuffer.push_back({f, target});
                syntheticGenerated++;
            }
        }

        if (replayBuffer.size() > replayCap) {
            size_t excess = replayBuffer.size() - replayCap;
            replayBuffer.erase(replayBuffer.begin(), replayBuffer.begin() + excess);
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            std::shuffle(replayBuffer.begin(), replayBuffer.end(), rng);
            for (auto& [features, target] : replayBuffer) {
                net.trainStep(features, target, learningRate);
            }
        }

        net.save(weightsPath);

        auto genEnd = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(genEnd - genStart).count();

        std::cout << "generation " << gen << "/" << generations
                  << " games=" << gamesPerGen
                  << " new_samples=" << newSamples
                  << " synthetic=" << syntheticGenerated
                  << " replay_buffer=" << replayBuffer.size()
                  << " W=" << whiteWins << " B=" << blackWins << " D=" << draws
                  << " time=" << secs << "s\n";
        std::cout.flush();
    }

    std::cout << "Training complete. Weights saved to " << weightsPath << "\n";
    return 0;
}
