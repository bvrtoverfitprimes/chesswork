#include "network.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <random>
#include <sstream>

namespace human_limit {

namespace {

constexpr int kMgPawn = 82, kEgPawn = 94;
constexpr int kMgKnight = 337, kEgKnight = 281;
constexpr int kMgBishop = 365, kEgBishop = 297;
constexpr int kMgRook = 477, kEgRook = 512;
constexpr int kMgQueen = 1025, kEgQueen = 936;

const int kMgPawnTable[64] = {
     0,   0,   0,   0,   0,   0,  0,   0,
    98, 134,  61,  95,  68, 126, 34, -11,
    -6,   7,  26,  31,  65,  56, 25, -20,
   -14,  13,   6,  21,  23,  12, 17, -23,
   -27,  -2,  -5,  12,  17,   6, 10, -25,
   -26,  -4,  -4, -10,   3,   3, 33, -12,
   -35,  -1, -20, -23, -15,  24, 38, -22,
     0,   0,   0,   0,   0,   0,  0,   0,
};
const int kEgPawnTable[64] = {
     0,   0,   0,   0,   0,   0,   0,   0,
   178, 173, 158, 134, 147, 132, 165, 187,
    94, 100,  85,  67,  56,  53,  82,  84,
    32,  24,  13,   5,  -2,   4,  17,  17,
    13,   9,  -3,  -7,  -7,  -8,   3,  -1,
     4,   7,  -6,   1,   0,  -5,  -1,  -8,
    13,   8,   8,  10,  13,   0,   2,  -7,
     0,   0,   0,   0,   0,   0,   0,   0,
};
const int kMgKnightTable[64] = {
  -167, -89, -34, -49,  61, -97, -15, -107,
   -73, -41,  72,  36,  23,  62,   7,  -17,
   -47,  60,  37,  65,  84, 129,  73,   44,
    -9,  17,  19,  53,  37,  69,  18,   22,
   -13,   4,  16,  13,  28,  19,  21,   -8,
   -23,  -9,  12,  10,  19,  17,  25,  -16,
   -29, -53, -12,  -3,  -1,  18, -14,  -19,
  -105, -21, -58, -33, -17, -28, -19,  -23,
};
const int kEgKnightTable[64] = {
   -58, -38, -13, -28, -31, -27, -63, -99,
   -25,  -8, -25,  -2,  -9, -25, -24, -52,
   -24, -20,  10,   9,  -1,  -9, -19, -41,
   -17,   3,  22,  22,  22,  11,   8, -18,
   -18,  -6,  16,  25,  16,  17,   4, -18,
   -23,  -3,  -1,  15,  10,  -3, -20, -22,
   -42, -20, -10,  -5,  -2, -20, -23, -44,
   -29, -51, -23, -15, -22, -18, -50, -64,
};
const int kMgBishopTable[64] = {
   -29,   4, -82, -37, -25, -42,   7,  -8,
   -26,  16, -18, -13,  30,  59,  18, -47,
   -16,  37,  43,  40,  35,  50,  37,  -2,
    -4,   5,  19,  50,  37,  37,   7,  -2,
    -6,  13,  13,  26,  34,  12,  10,   4,
     0,  15,  15,  15,  14,  27,  18,  10,
     4,  15,  16,   0,   7,  21,  33,   1,
   -33,  -3, -14, -21, -13, -12, -39, -21,
};
const int kEgBishopTable[64] = {
   -14, -21, -11,  -8,  -7,  -9, -17, -24,
    -8,  -4,   7, -12,  -3, -13,  -4, -14,
     2,  -8,   0,  -1,  -2,   6,   0,   4,
    -3,   9,  12,   9,  14,  10,   3,   2,
    -6,   3,  13,  19,   7,  10,  -3,  -9,
   -12,  -3,   8,  10,  13,   3,  -7, -15,
   -14, -18,  -7,  -1,   4,  -9, -15, -27,
   -23,  -9, -23,  -5,  -9, -16,  -5, -17,
};
const int kMgRookTable[64] = {
    32,  42,  32,  51, 63,  9,  31,  43,
    27,  32,  58,  62, 80, 67,  26,  44,
    -5,  19,  26,  36, 17, 45,  61,  16,
   -24, -11,   7,  26, 24, 35,  -8, -20,
   -36, -26, -12,  -1,  9, -7,   6, -23,
   -45, -25, -16, -17,  3,  0,  -5, -33,
   -44, -16, -20,  -9, -1, 11,  -6, -71,
   -19, -13,   1,  17, 16,  7, -37, -26,
};
const int kEgRookTable[64] = {
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
};
const int kMgQueenTable[64] = {
   -28,   0,  29,  12,  59,  44,  43,  45,
   -24, -39,  -5,   1, -16,  57,  28,  54,
   -13, -17,   7,   8,  29,  56,  47,  57,
   -27, -27, -16, -16,  -1,  17,  -2,   1,
    -9, -26,  -9, -10,  -2,  -4,   3,  -3,
   -14,   2, -11,  -2,  -5,   2,  14,   5,
   -35,  -8,  11,   2,   8,  15,  -3,   1,
    -1, -18,  -9,  10, -15, -25, -31, -50,
};
const int kEgQueenTable[64] = {
    -9,  22,  22,  27,  27,  19,  10,  20,
   -17,  20,  32,  41,  58,  25,  30,   0,
   -20,   6,   9,  49,  47,  35,  19,   9,
     3,  22,  24,  45,  57,  40,  57,  36,
   -18,  28,  19,  47,  31,  34,  39,  23,
   -16, -27,  15,   6,   9,  17,  10,   5,
   -22, -23, -30, -16, -16, -23, -36, -32,
   -33, -28, -22, -43,  -5, -32, -20, -41,
};
const int kMgKingTable[64] = {
   -65,  23,  16, -15, -56, -34,   2,  13,
    29,  -1, -20,  -7,  -8,  -4, -38, -29,
    -9,  24,   2, -16, -20,   6,  22, -22,
   -17, -20, -12, -27, -30, -25, -14, -36,
   -49,  -1, -27, -39, -46, -44, -33, -51,
   -14, -14, -22, -46, -44, -30, -15, -27,
     1,   7,  -8, -64, -43, -16,   9,   8,
   -15,  36,  12, -54,   8, -28,  24,  14,
};
const int kEgKingTable[64] = {
   -74, -35, -18, -18, -11,  15,   4, -17,
   -12,  17,  14,  17,  17,  38,  23,  11,
    10,  17,  23,  15,  20,  45,  44,  13,
    -8,  22,  24,  27,  26,  33,  26,   3,
   -18,  -4,  21,  24,  27,  23,   9, -11,
   -19,  -3,  11,  21,  23,  16,   7,  -9,
   -27, -11,   4,  13,  14,   4,  -5, -17,
   -53, -34, -21, -11, -28, -14, -24, -43,
};

struct PieceTables {
    int mgMaterial;
    int egMaterial;
    const int* mgTable;
    const int* egTable;
    int phaseWeight;
};

const PieceTables kPawnT{kMgPawn, kEgPawn, kMgPawnTable, kEgPawnTable, 0};
const PieceTables kKnightT{kMgKnight, kEgKnight, kMgKnightTable, kEgKnightTable, 1};
const PieceTables kBishopT{kMgBishop, kEgBishop, kMgBishopTable, kEgBishopTable, 1};
const PieceTables kRookT{kMgRook, kEgRook, kMgRookTable, kEgRookTable, 2};
const PieceTables kQueenT{kMgQueen, kEgQueen, kMgQueenTable, kEgQueenTable, 4};
const PieceTables kKingT{0, 0, kMgKingTable, kEgKingTable, 0};

const PieceTables& tablesFor(char pl) {
    switch (pl) {
        case 'p': return kPawnT;
        case 'n': return kKnightT;
        case 'b': return kBishopT;
        case 'r': return kRookT;
        case 'q': return kQueenT;
        default: return kKingT;
    }
}

double relu(double v) { return v > 0.0 ? v : 0.0; }
double reluDeriv(double v) { return v > 0.0 ? 1.0 : 0.0; }

}

Features extractFeatures(const chess::Game& gameConst) {
    chess::Game& game = const_cast<chess::Game&>(gameConst);
    const auto& board = game.boardArray();

    int matDiff[5] = {0, 0, 0, 0, 0};
    int mg = 0, eg = 0, phase = 0;
    chess::Pos whiteKing{-1, -1};
    chess::Pos blackKing{-1, -1};

    int whitePawnsByFile[8] = {0};
    int blackPawnsByFile[8] = {0};
    int whiteBishops = 0, blackBishops = 0;
    int whiteAdvancedKnights = 0, blackAdvancedKnights = 0;
    int whiteCentralPawns = 0, blackCentralPawns = 0;

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;
            char pl = std::tolower(static_cast<unsigned char>(p));
            bool white = std::isupper(static_cast<unsigned char>(p));
            const PieceTables& t = tablesFor(pl);
            int tableIndex = white ? (r * 8 + c) : ((7 - r) * 8 + c);
            int mgVal = t.mgMaterial + t.mgTable[tableIndex];
            int egVal = t.egMaterial + t.egTable[tableIndex];

            if (white) { mg += mgVal; eg += egVal; } else { mg -= mgVal; eg -= egVal; }
            phase += t.phaseWeight;

            int idx = -1;
            switch (pl) {
                case 'p':
                    idx = 0;
                    if (white) whitePawnsByFile[c]++; else blackPawnsByFile[c]++;
                    if ((r == 3 || r == 4) && (c == 3 || c == 4)) { if (white) whiteCentralPawns++; else blackCentralPawns++; }
                    break;
                case 'n':
                    idx = 1;
                    if (white && r <= 3) whiteAdvancedKnights++;
                    if (!white && r >= 4) blackAdvancedKnights++;
                    break;
                case 'b':
                    idx = 2;
                    if (white) whiteBishops++; else blackBishops++;
                    break;
                case 'r': idx = 3; break;
                case 'q': idx = 4; break;
                case 'k':
                    if (white) whiteKing = {r, c}; else blackKing = {r, c};
                    break;
            }
            if (idx >= 0) matDiff[idx] += white ? 1 : -1;
        }
    }
    if (phase > 24) phase = 24;

    int whiteDoubled = 0, blackDoubled = 0;
    int whiteIsolated = 0, blackIsolated = 0;
    for (int f = 0; f < 8; f++) {
        if (whitePawnsByFile[f] > 1) whiteDoubled += whitePawnsByFile[f] - 1;
        if (blackPawnsByFile[f] > 1) blackDoubled += blackPawnsByFile[f] - 1;
        bool whiteLeftNeighbor = (f > 0 && whitePawnsByFile[f - 1] > 0);
        bool whiteRightNeighbor = (f < 7 && whitePawnsByFile[f + 1] > 0);
        if (whitePawnsByFile[f] > 0 && !whiteLeftNeighbor && !whiteRightNeighbor) whiteIsolated += whitePawnsByFile[f];
        bool blackLeftNeighbor = (f > 0 && blackPawnsByFile[f - 1] > 0);
        bool blackRightNeighbor = (f < 7 && blackPawnsByFile[f + 1] > 0);
        if (blackPawnsByFile[f] > 0 && !blackLeftNeighbor && !blackRightNeighbor) blackIsolated += blackPawnsByFile[f];
    }

    int whitePassed = 0, blackPassed = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == 'P') {
                bool blocked = false;
                for (int rr = 0; rr < r && !blocked; rr++) {
                    for (int cc = std::max(0, c - 1); cc <= std::min(7, c + 1); cc++) {
                        if (board[rr][cc] == 'p') { blocked = true; break; }
                    }
                }
                if (!blocked) whitePassed++;
            } else if (p == 'p') {
                bool blocked = false;
                for (int rr = r + 1; rr < 8 && !blocked; rr++) {
                    for (int cc = std::max(0, c - 1); cc <= std::min(7, c + 1); cc++) {
                        if (board[rr][cc] == 'P') { blocked = true; break; }
                    }
                }
                if (!blocked) blackPassed++;
            }
        }
    }

    int whiteOpenRooks = 0, blackOpenRooks = 0, whiteSemiOpenRooks = 0, blackSemiOpenRooks = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == 'R') {
                if (whitePawnsByFile[c] == 0 && blackPawnsByFile[c] == 0) whiteOpenRooks++;
                else if (whitePawnsByFile[c] == 0) whiteSemiOpenRooks++;
            } else if (p == 'r') {
                if (whitePawnsByFile[c] == 0 && blackPawnsByFile[c] == 0) blackOpenRooks++;
                else if (blackPawnsByFile[c] == 0) blackSemiOpenRooks++;
            }
        }
    }

    bool whiteKingExposed = whiteKing.r >= 0 && whitePawnsByFile[whiteKing.c] == 0;
    bool blackKingExposed = blackKing.r >= 0 && blackPawnsByFile[blackKing.c] == 0;

    int whiteKingAttacks = 0, blackKingAttacks = 0;
    const std::array<std::pair<int, int>, 8> neighbors = {{
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    }};
    if (whiteKing.r >= 0) {
        for (auto [dr, dc] : neighbors) {
            int r = whiteKing.r + dr, c = whiteKing.c + dc;
            if (chess::onBoard(r, c) && chess::isAttacked(board, r, c, chess::Color::White)) whiteKingAttacks++;
        }
    }
    if (blackKing.r >= 0) {
        for (auto [dr, dc] : neighbors) {
            int r = blackKing.r + dr, c = blackKing.c + dc;
            if (chess::onBoard(r, c) && chess::isAttacked(board, r, c, chess::Color::Black)) blackKingAttacks++;
        }
    }

    chess::Color sideToMove = game.turn();
    int mobilitySide = static_cast<int>(game.getValidMovesUci().size());
    auto nullUndo = game.makeNullMove();
    int mobilityOther = static_cast<int>(game.getValidMovesUci().size());
    game.unmakeNullMove(nullUndo);

    int whiteMobility = (sideToMove == chess::Color::White) ? mobilitySide : mobilityOther;
    int blackMobility = (sideToMove == chess::Color::Black) ? mobilitySide : mobilityOther;

    Features f;
    f.x[0] = matDiff[0] / 8.0;
    f.x[1] = matDiff[1] / 4.0;
    f.x[2] = matDiff[2] / 4.0;
    f.x[3] = matDiff[3] / 4.0;
    f.x[4] = matDiff[4] / 2.0;
    f.x[5] = mg / 500.0;
    f.x[6] = eg / 500.0;
    f.x[7] = phase / 24.0;
    f.x[8] = (whiteMobility - blackMobility) / 20.0;
    f.x[9] = (blackKingAttacks - whiteKingAttacks) / 8.0;
    f.x[10] = (blackDoubled - whiteDoubled) / 8.0;
    f.x[11] = (blackIsolated - whiteIsolated) / 8.0;
    f.x[12] = (whitePassed - blackPassed) / 8.0;
    f.x[13] = static_cast<double>((whiteBishops >= 2 ? 1 : 0) - (blackBishops >= 2 ? 1 : 0));
    f.x[14] = (whiteOpenRooks - blackOpenRooks) / 4.0;
    f.x[15] = (whiteSemiOpenRooks - blackSemiOpenRooks) / 4.0;
    f.x[16] = (whiteAdvancedKnights - blackAdvancedKnights) / 4.0;
    f.x[17] = static_cast<double>((blackKingExposed ? 1 : 0) - (whiteKingExposed ? 1 : 0));
    f.x[18] = (whiteCentralPawns - blackCentralPawns) / 4.0;
    f.x[19] = (sideToMove == chess::Color::White) ? 1.0 : -1.0;
    f.classicalEval = (mg * phase + eg * (24 - phase)) / 24.0;
    return f;
}

Network::Network() { randomInit(); }

void Network::randomInit() {
    std::mt19937 rng(42);
    double s1 = std::sqrt(2.0 / kInputSize);
    double s2 = std::sqrt(2.0 / kHidden1Size);
    double s3 = std::sqrt(2.0 / kHidden2Size);
    std::normal_distribution<double> d1(0.0, s1);
    std::normal_distribution<double> d2(0.0, s2);
    std::normal_distribution<double> d3(0.0, s3);

    for (auto& row : w1_) for (auto& v : row) v = d1(rng);
    for (auto& v : b1_) v = 0.0;
    for (auto& row : w2_) for (auto& v : row) v = d2(rng);
    for (auto& v : b2_) v = 0.0;
    for (auto& row : w3_) for (auto& v : row) v = d3(rng);
    for (auto& v : b3_) v = 0.0;
    for (auto& v : w4_) v = d3(rng);
    b4_ = 0.0;
}

double Network::forward(const Features& f, std::array<double, kHidden1Size>* h1Out,
                         std::array<double, kHidden2Size>* h2Out,
                         std::array<double, kHidden3Size>* h3Out) const {
    std::array<double, kHidden1Size> h1;
    for (int i = 0; i < kHidden1Size; i++) {
        double s = b1_[i];
        for (int j = 0; j < kInputSize; j++) s += w1_[i][j] * f.x[j];
        h1[i] = relu(s);
    }
    std::array<double, kHidden2Size> h2;
    for (int i = 0; i < kHidden2Size; i++) {
        double s = b2_[i];
        for (int j = 0; j < kHidden1Size; j++) s += w2_[i][j] * h1[j];
        h2[i] = relu(s);
    }
    std::array<double, kHidden3Size> h3;
    for (int i = 0; i < kHidden3Size; i++) {
        double s = b3_[i];
        for (int j = 0; j < kHidden2Size; j++) s += w3_[i][j] * h2[j];
        h3[i] = relu(s);
    }
    double out = b4_;
    for (int j = 0; j < kHidden3Size; j++) out += w4_[j] * h3[j];

    if (h1Out) *h1Out = h1;
    if (h2Out) *h2Out = h2;
    if (h3Out) *h3Out = h3;
    return out;
}

double Network::evaluate(const chess::Game& game) const {
    Features f = extractFeatures(game);
    double networkScore = forward(f) * 1000.0;
    return kClassicalWeight * f.classicalEval + kNetworkWeight * networkScore;
}

void Network::trainStep(const Features& f, double target, double learningRate) {
    std::array<double, kHidden1Size> h1;
    std::array<double, kHidden2Size> h2;
    std::array<double, kHidden3Size> h3;
    double pred = forward(f, &h1, &h2, &h3);

    double dOut = 2.0 * (pred - target);

    std::array<double, kHidden3Size> dH3;
    for (int j = 0; j < kHidden3Size; j++) {
        dH3[j] = dOut * w4_[j] * reluDeriv(h3[j]);
    }
    std::array<double, kHidden2Size> dH2;
    for (int j = 0; j < kHidden2Size; j++) {
        double s = 0.0;
        for (int i = 0; i < kHidden3Size; i++) s += dH3[i] * w3_[i][j];
        dH2[j] = s * reluDeriv(h2[j]);
    }
    std::array<double, kHidden1Size> dH1;
    for (int j = 0; j < kHidden1Size; j++) {
        double s = 0.0;
        for (int i = 0; i < kHidden2Size; i++) s += dH2[i] * w2_[i][j];
        dH1[j] = s * reluDeriv(h1[j]);
    }

    for (int j = 0; j < kHidden3Size; j++) w4_[j] -= learningRate * dOut * h3[j];
    b4_ -= learningRate * dOut;

    for (int i = 0; i < kHidden3Size; i++) {
        for (int j = 0; j < kHidden2Size; j++) w3_[i][j] -= learningRate * dH3[i] * h2[j];
        b3_[i] -= learningRate * dH3[i];
    }

    for (int i = 0; i < kHidden2Size; i++) {
        for (int j = 0; j < kHidden1Size; j++) w2_[i][j] -= learningRate * dH2[i] * h1[j];
        b2_[i] -= learningRate * dH2[i];
    }

    for (int i = 0; i < kHidden1Size; i++) {
        for (int j = 0; j < kInputSize; j++) w1_[i][j] -= learningRate * dH1[i] * f.x[j];
        b1_[i] -= learningRate * dH1[i];
    }
}

void Network::save(const std::string& path) const {
    std::ofstream out(path);
    for (auto& row : w1_) for (auto v : row) out << v << " ";
    for (auto v : b1_) out << v << " ";
    for (auto& row : w2_) for (auto v : row) out << v << " ";
    for (auto v : b2_) out << v << " ";
    for (auto& row : w3_) for (auto v : row) out << v << " ";
    for (auto v : b3_) out << v << " ";
    for (auto v : w4_) out << v << " ";
    out << b4_ << "\n";
}

bool Network::load(const std::string& path) {
    std::ifstream in(path);
    if (!in) return false;
    for (auto& row : w1_) for (auto& v : row) if (!(in >> v)) return false;
    for (auto& v : b1_) if (!(in >> v)) return false;
    for (auto& row : w2_) for (auto& v : row) if (!(in >> v)) return false;
    for (auto& v : b2_) if (!(in >> v)) return false;
    for (auto& row : w3_) for (auto& v : row) if (!(in >> v)) return false;
    for (auto& v : b3_) if (!(in >> v)) return false;
    for (auto& v : w4_) if (!(in >> v)) return false;
    if (!(in >> b4_)) return false;
    return true;
}

}
