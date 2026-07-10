#pragma once

#include "../../chess/bitboard/position.h"

namespace raw_engine {

// per-term attribution (white-relative cp, already phase-tapered).
// Terms not yet implemented stay 0 -- a permanently-zero row in a blunder
// diagnosis is itself a signal that a term is MISSING, not just mistuned.
struct EvalBreakdown {
    int material = 0;
    int pst = 0;

    int mobility = 0;
    int kingSafety = 0;
    int kingAttack = 0;

    int pawnStructure = 0;
    int passedPawns = 0;
    int pawnShield = 0;

    int rookFiles = 0;
    int rookSeventh = 0;
    int bishopPair = 0;
    int bishopMobility = 0;
    int knightOutposts = 0;

    int threats = 0;
    int pins = 0;
    int hangingPieces = 0;
    int tacticalPressure = 0;

    int pieceQuality = 0;  // king tropism w/ obstruction discount
    int space = 0;
    int centerControl = 0;
    int endgameKing = 0;
    int tempo = 0;
    int phase = 0;
    int total = 0;
};

int evaluateWhiteRelative(const chess::bitboard::Position& pos, EvalBreakdown* bd = nullptr);

// prints per-piece effective-value explanation (tropism, outpost, mobility) to stdout
void printPieceQuality(const chess::bitboard::Position& pos);

}
