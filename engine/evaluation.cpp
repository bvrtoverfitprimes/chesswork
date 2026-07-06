#include "evaluation.h"

#include <cctype>

namespace engine {

namespace {

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

bool isCenterSquare(int r, int c) {
    return (r == 3 || r == 4) && (c == 3 || c == 4);
}

} // namespace

int evaluate(const chess::BoardArray& board) {
    int score = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;

            char pl = std::tolower(static_cast<unsigned char>(p));
            int value = pieceValue(pl);
            if (pl != 'k' && isCenterSquare(r, c)) value += 10;

            if (std::isupper(static_cast<unsigned char>(p))) score += value;
            else score -= value;
        }
    }
    return score;
}

} // namespace engine
