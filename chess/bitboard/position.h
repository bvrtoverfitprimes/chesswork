#pragma once

#include <array>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "../pieces.h"
#include "bitboard.h"

namespace chess::bitboard {

enum class MoveType { Normal, Castle, EnPassant, Promote };

struct BBMove {
    int from;
    int to;
    char promotion = ' ';

    bool operator==(const BBMove& o) const { return from == o.from && to == o.to && promotion == o.promotion; }
};

struct BBUndo {
    int from = 0;
    int to = 0;
    char piece = ' ';
    char captured = ' ';
    std::array<bool, 4> prevCastlingRights{};
    std::optional<int> prevEnPassant;
    uint64_t prevHash = 0;
    int prevHalfMoves = 0;
    MoveType moveType = MoveType::Normal;

    int rookFrom = -1;
    int rookTo = -1;
    int epCapturedSquare = -1;
    char epCapturedPiece = ' ';
    char promoteFrom = ' ';
};

class Position {
public:
    Position();
    explicit Position(const std::string& fen);

    BBUndo makeMove(int from, int to, char promotion = 'q');
    void unmakeMove(const BBUndo& undo);

    struct NullUndo {
        std::optional<int> prevEnPassant;
        uint64_t prevHash;
    };
    NullUndo makeNullMove();
    void unmakeNullMove(const NullUndo& undo);

    std::vector<BBMove> getValidMoves();

    chess::Color turn() const { return turn_; }
    uint64_t zobristHash() const { return zobristHash_; }
    char pieceAt(int sq) const { return mailbox_[sq]; }
    std::optional<int> enPassantTarget() const { return enPassantTarget_; }
    Bitboard pawnBitboard(chess::Color c) const { return pieces_[c == chess::Color::White ? 0 : 1][0]; }
    Bitboard nonPawnBitboard(chess::Color c) const {
        int s = c == chess::Color::White ? 0 : 1;
        return pieces_[s][1] | pieces_[s][2] | pieces_[s][3] | pieces_[s][4];
    }
    bool hasNonPawnMaterial(chess::Color c) const {
        int side = (c == chess::Color::White) ? 0 : 1;
        return (pieces_[side][1] | pieces_[side][2] | pieces_[side][3] | pieces_[side][4]) != 0;
    }
    bool inCheck() const;
    bool isFiftyMoveDraw() const { return halfMoves_ >= 100; }
    bool isRepetitionDraw() const;
    bool isInsufficientMaterial() const;

    chess::BoardArray toBoardArray() const;
    std::string toFen() const;

    Bitboard attackersTo(int sq, Bitboard occ) const;
    int see(int from, int to) const;

    static int parseSquareUci(const std::string& s);
    static std::string squareToUci(int sq);

private:
    std::array<char, 64> mailbox_{};
    std::array<Bitboard, 2> occupied_{};
    Bitboard allOccupied_ = 0;
    std::array<std::array<Bitboard, 6>, 2> pieces_{};

    chess::Color turn_ = chess::Color::White;
    std::array<bool, 4> castlingRights_{true, true, true, true};
    std::optional<int> enPassantTarget_;
    int halfMoves_ = 0;
    int whiteKingSq_ = -1;
    int blackKingSq_ = -1;

    std::array<std::array<uint64_t, 13>, 64> zobristTable_{};
    uint64_t zobristBlackToMoveKey_ = 0;
    std::array<uint64_t, 4> zobristCastling_{};
    std::array<uint64_t, 8> zobristEnPassant_{};
    uint64_t zobristHash_ = 0;
    std::map<uint64_t, int> repetitionTable_;

    void setupFromMailbox();
    void setPiece(int sq, char piece);
    void clearPiece(int sq);
    void initZobrist();
    void resetRepetitionTable();

    bool squareAttackedBy(int sq, chess::Color attacker) const;
    int seeSwing(Bitboard occ, int sq, bool attackerIsWhite, int standValue) const;
    static int pieceTypeIdx(char pieceLower);
    static int zobristPieceIndex(char piece);
};

}
