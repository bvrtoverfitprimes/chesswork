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

    std::vector<std::string> getValidMovesUci();

    Color turn() const { return turn_; }
    bool inCheck() const;
    bool isCheckmate();
    bool isStalemate();
    bool isRepetitionDraw() const;
    bool isFiftyMoveDraw() const;
    bool isInsufficientMaterial() const;

    static Pos parseSquare(const std::string& s);
    static std::string squareToStr(Pos p);

private:
    BoardArray board_;
    Color turn_ = Color::White;
    CastlingRights castlingRights_ = {true, true, true, true};
    std::optional<Pos> enPassantTarget_;
    int halfMoves_ = 0;

    std::array<std::array<uint64_t, 13>, 64> zobristTable_;
    uint64_t zobristHash_ = 0;
    std::map<uint64_t, int> repetitionTable_;

    void initZobrist();
    void resetRepetitionTable();
    static int pieceIndex(char piece);
};

} // namespace chess
