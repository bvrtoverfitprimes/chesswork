#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>

#include "pieces.h"

namespace chess {

enum class MoveType { Normal, Castle, EnPassant, Promote };

struct UndoMove {
    Pos from;
    Pos to;
    char piece;
    char captured;
    CastlingRights prevCastlingRights;
    std::optional<Pos> prevEnPassant;
    uint64_t prevHash;
    int prevHalfMoves;
    MoveType moveType;

    Pos rookFrom;
    Pos rookTo;

    Pos epCapturedSquare;
    char epCapturedPiece;

    char promoteFrom;
};

class Game {
public:
    Game();
    explicit Game(const std::string& fen);

    void printBoard() const;

    void run();

    UndoMove makeMove(Pos from, Pos to, char promotion = 'q');
    void unmakeMove(const UndoMove& undo);

    struct NullUndo {
        std::optional<Pos> prevEnPassant;
        uint64_t prevHash;
    };
    NullUndo makeNullMove();
    void unmakeNullMove(const NullUndo& undo);

    std::vector<std::string> getValidMovesUci();
    // Same legality filtering as getValidMovesUci(), without the UCI-string round-trip —
    // for hot-path callers (search) that already work with Move structs.
    std::vector<Move> getValidMoves();
    // Kept permanently as a slow, structurally-simple oracle for tests/test_fast_legality.cpp
    // to cross-validate getValidMovesUci()'s pin/check-mask fast path against.
    std::vector<std::string> getValidMovesUciSlow();

    Color turn() const { return turn_; }
    const BoardArray& boardArray() const { return board_; }
    uint64_t zobristHash() const { return zobristHash_; }
    bool inCheck() const;
    bool isCheckmate();
    bool isStalemate();
    bool isRepetitionDraw() const;
    bool isFiftyMoveDraw() const;
    bool isInsufficientMaterial() const;

    static Pos parseSquare(const std::string& s);
    static std::string squareToStr(Pos p);
    std::string toFen() const;

private:
    BoardArray board_;
    Color turn_ = Color::White;
    CastlingRights castlingRights_ = {true, true, true, true};
    std::optional<Pos> enPassantTarget_;
    int halfMoves_ = 0;
    Pos whiteKing_{-1, -1};
    Pos blackKing_{-1, -1};

    std::array<std::array<uint64_t, 13>, 64> zobristTable_;
    uint64_t zobristBlackToMoveKey_ = 0;
    uint64_t zobristHash_ = 0;
    std::map<uint64_t, int> repetitionTable_;

    void initZobrist();
    void resetRepetitionTable();
    void findKings();
    bool inCheckColor(Color c) const;
    static int pieceIndex(char piece);
};

}
