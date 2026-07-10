#include "accumulator.h"

#include <cctype>

#include "nnue_features.h"

namespace limit {

namespace {

void recomputeSide(const Network& net, const chess::BoardArray& board, bool isWhite, std::vector<float>* out) {
    int hidden = net.hiddenSize();
    out->assign(hidden, 0.0f);
    PerspectiveContext ctx = computePerspectiveContext(board, isWhite);
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int idx = featureIndexForPiece(ctx, r, c, board[r][c]);
            if (idx < 0) continue;
            const float* row = net.embeddingRow(idx);
            for (int h = 0; h < hidden; h++) (*out)[h] += row[h];
        }
    }
}

void addRow(const Network& net, int idx, std::vector<float>* acc) {
    if (idx < 0) return;
    const float* row = net.embeddingRow(idx);
    int hidden = net.hiddenSize();
    for (int h = 0; h < hidden; h++) (*acc)[h] += row[h];
}

void subRow(const Network& net, int idx, std::vector<float>* acc) {
    if (idx < 0) return;
    const float* row = net.embeddingRow(idx);
    int hidden = net.hiddenSize();
    for (int h = 0; h < hidden; h++) (*acc)[h] -= row[h];
}

struct SquareChange {
    int r, c;
    char oldPiece;
    char newPiece;
};

void diffSide(const Network& net, const chess::BoardArray& boardAfter, bool isWhite,
              const std::vector<SquareChange>& changes, std::vector<float>* acc) {
    PerspectiveContext ctx = computePerspectiveContext(boardAfter, isWhite);
    for (const auto& ch : changes) {
        subRow(net, featureIndexForPiece(ctx, ch.r, ch.c, ch.oldPiece), acc);
        addRow(net, featureIndexForPiece(ctx, ch.r, ch.c, ch.newPiece), acc);
    }
}

}

namespace {

int countPieces(const chess::BoardArray& board) {
    int count = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] != ' ') count++;
        }
    }
    return count;
}

}

void initAccumulator(const Network& net, const chess::BoardArray& board, Accumulator* acc) {
    recomputeSide(net, board, true, &acc->white);
    recomputeSide(net, board, false, &acc->black);
    acc->pieceCount = countPieces(board);
}

void applyMoveToAccumulator(const Network& net, const chess::BoardArray& boardAfter, const chess::UndoMove& undo,
                             const Accumulator& before, Accumulator* after) {
    std::vector<SquareChange> changes;
    changes.reserve(4);

    char newAtTo = boardAfter[undo.to.r][undo.to.c];
    changes.push_back({undo.from.r, undo.from.c, undo.piece, ' '});
    changes.push_back({undo.to.r, undo.to.c, undo.captured, newAtTo});

    bool whiteKingMoved = (undo.piece == 'K');
    bool blackKingMoved = (undo.piece == 'k');

    if (undo.moveType == chess::MoveType::Castle) {
        char rookChar = whiteKingMoved ? 'R' : 'r';
        changes.push_back({undo.rookFrom.r, undo.rookFrom.c, rookChar, ' '});
        changes.push_back({undo.rookTo.r, undo.rookTo.c, ' ', rookChar});
    } else if (undo.moveType == chess::MoveType::EnPassant) {
        changes.push_back({undo.epCapturedSquare.r, undo.epCapturedSquare.c, undo.epCapturedPiece, ' '});
    }

    bool captured = (undo.captured != ' ') || (undo.moveType == chess::MoveType::EnPassant);

    if (&before != after) *after = before;
    if (captured) after->pieceCount = before.pieceCount - 1;

    if (whiteKingMoved) {
        recomputeSide(net, boardAfter, true, &after->white);
    } else {
        diffSide(net, boardAfter, true, changes, &after->white);
    }

    if (blackKingMoved) {
        recomputeSide(net, boardAfter, false, &after->black);
    } else {
        diffSide(net, boardAfter, false, changes, &after->black);
    }
}

namespace {

chess::Pos fromLerfSquare(int sq) {
    return {7 - chess::bitboard::rankOf(sq), chess::bitboard::fileOf(sq)};
}

}

void initAccumulator(const Network& net, const chess::bitboard::Position& pos, Accumulator* acc) {
    initAccumulator(net, pos.toBoardArray(), acc);
}

void applyMoveToAccumulator(const Network& net, const chess::bitboard::Position& posAfter,
                             const chess::bitboard::BBUndo& undo, const Accumulator& before, Accumulator* after) {
    chess::BoardArray boardAfter = posAfter.toBoardArray();

    std::vector<SquareChange> changes;
    changes.reserve(4);

    chess::Pos from = fromLerfSquare(undo.from);
    chess::Pos to = fromLerfSquare(undo.to);
    char newAtTo = boardAfter[to.r][to.c];
    changes.push_back({from.r, from.c, undo.piece, ' '});
    changes.push_back({to.r, to.c, undo.captured, newAtTo});

    bool whiteKingMoved = (undo.piece == 'K');
    bool blackKingMoved = (undo.piece == 'k');

    if (undo.moveType == chess::bitboard::MoveType::Castle) {
        char rookChar = whiteKingMoved ? 'R' : 'r';
        chess::Pos rookFrom = fromLerfSquare(undo.rookFrom);
        chess::Pos rookTo = fromLerfSquare(undo.rookTo);
        changes.push_back({rookFrom.r, rookFrom.c, rookChar, ' '});
        changes.push_back({rookTo.r, rookTo.c, ' ', rookChar});
    } else if (undo.moveType == chess::bitboard::MoveType::EnPassant) {
        chess::Pos epSq = fromLerfSquare(undo.epCapturedSquare);
        changes.push_back({epSq.r, epSq.c, undo.epCapturedPiece, ' '});
    }

    bool captured = (undo.captured != ' ') || (undo.moveType == chess::bitboard::MoveType::EnPassant);

    if (&before != after) *after = before;
    if (captured) after->pieceCount = before.pieceCount - 1;

    if (whiteKingMoved) {
        recomputeSide(net, boardAfter, true, &after->white);
    } else {
        diffSide(net, boardAfter, true, changes, &after->white);
    }

    if (blackKingMoved) {
        recomputeSide(net, boardAfter, false, &after->black);
    } else {
        diffSide(net, boardAfter, false, changes, &after->black);
    }
}

}
