#pragma once

#include <vector>

#include "../../chess/bitboard/position.h"
#include "../../chess/board.h"
#include "network.h"

namespace human_limit {

struct Accumulator {
    std::vector<float> white;
    std::vector<float> black;
    int pieceCount = 0;
};

void initAccumulator(const Network& net, const chess::BoardArray& board, Accumulator* acc);

void applyMoveToAccumulator(const Network& net, const chess::BoardArray& boardAfter, const chess::UndoMove& undo,
                             const Accumulator& before, Accumulator* after);

void initAccumulator(const Network& net, const chess::bitboard::Position& pos, Accumulator* acc);
void applyMoveToAccumulator(const Network& net, const chess::bitboard::Position& posAfter,
                             const chess::bitboard::BBUndo& undo, const Accumulator& before, Accumulator* after);

}
