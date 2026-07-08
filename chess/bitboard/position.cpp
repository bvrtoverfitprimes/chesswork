#include "position.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <sstream>

#include "magic.h"

namespace chess::bitboard {

namespace {

std::mt19937_64 zobristRng(0xB17B0A2D5EEDULL);

bool isUpperPiece(char p) { return p != ' ' && std::isupper(static_cast<unsigned char>(p)); }

constexpr int kWhiteKingStart = 4;    // e1
constexpr int kWhiteRookAStart = 0;   // a1
constexpr int kWhiteRookHStart = 7;   // h1
constexpr int kBlackKingStart = 60;   // e8
constexpr int kBlackRookAStart = 56;  // a8
constexpr int kBlackRookHStart = 63;  // h8

}

int Position::pieceTypeIdx(char pieceLower) {
    switch (pieceLower) {
        case 'p': return 0;
        case 'n': return 1;
        case 'b': return 2;
        case 'r': return 3;
        case 'q': return 4;
        case 'k': return 5;
        default: return -1;
    }
}

int Position::zobristPieceIndex(char piece) {
    switch (piece) {
        case 'P': return 1; case 'N': return 2; case 'B': return 3; case 'R': return 4; case 'Q': return 5; case 'K': return 6;
        case 'p': return 7; case 'n': return 8; case 'b': return 9; case 'r': return 10; case 'q': return 11; case 'k': return 12;
        default: return 0;
    }
}

void Position::setPiece(int sq, char piece) {
    mailbox_[sq] = piece;
    if (piece == ' ') return;
    bool white = isUpperPiece(piece);
    int pt = pieceTypeIdx(std::tolower(static_cast<unsigned char>(piece)));
    pieces_[white ? 0 : 1][pt] |= (1ULL << sq);
    occupied_[white ? 0 : 1] |= (1ULL << sq);
    allOccupied_ |= (1ULL << sq);
}

void Position::clearPiece(int sq) {
    char piece = mailbox_[sq];
    if (piece == ' ') return;
    bool white = isUpperPiece(piece);
    int pt = pieceTypeIdx(std::tolower(static_cast<unsigned char>(piece)));
    pieces_[white ? 0 : 1][pt] &= ~(1ULL << sq);
    occupied_[white ? 0 : 1] &= ~(1ULL << sq);
    allOccupied_ &= ~(1ULL << sq);
    mailbox_[sq] = ' ';
}

void Position::setupFromMailbox() {
    pieces_ = {};
    occupied_ = {};
    allOccupied_ = 0;
    whiteKingSq_ = -1;
    blackKingSq_ = -1;
    for (int sq = 0; sq < 64; sq++) {
        char piece = mailbox_[sq];
        if (piece == ' ') continue;
        bool white = isUpperPiece(piece);
        int pt = pieceTypeIdx(std::tolower(static_cast<unsigned char>(piece)));
        pieces_[white ? 0 : 1][pt] |= (1ULL << sq);
        occupied_[white ? 0 : 1] |= (1ULL << sq);
        allOccupied_ |= (1ULL << sq);
        if (piece == 'K') whiteKingSq_ = sq;
        if (piece == 'k') blackKingSq_ = sq;
    }
}

void Position::initZobrist() {
    for (auto& row : zobristTable_)
        for (auto& v : row) v = zobristRng();
    zobristBlackToMoveKey_ = zobristRng();
    for (auto& v : zobristCastling_) v = zobristRng();
    for (auto& v : zobristEnPassant_) v = zobristRng();

    uint64_t h = 0;
    for (int sq = 0; sq < 64; sq++) {
        char p = mailbox_[sq];
        if (p != ' ') h ^= zobristTable_[sq][zobristPieceIndex(p)];
    }
    if (turn_ == chess::Color::Black) h ^= zobristBlackToMoveKey_;
    for (int i = 0; i < 4; i++) {
        if (castlingRights_[i]) h ^= zobristCastling_[i];
    }
    if (enPassantTarget_) h ^= zobristEnPassant_[fileOf(*enPassantTarget_)];
    zobristHash_ = h;
}

void Position::resetRepetitionTable() {
    repetitionTable_.clear();
    repetitionTable_[zobristHash_] = 1;
}

Position::Position() {
    mailbox_.fill(' ');
    for (int f = 0; f < 8; f++) {
        mailbox_[squareOf(f, 1)] = 'P';
        mailbox_[squareOf(f, 6)] = 'p';
    }
    const char backRank[8] = {'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'};
    for (int f = 0; f < 8; f++) {
        mailbox_[squareOf(f, 0)] = backRank[f];
        mailbox_[squareOf(f, 7)] = static_cast<char>(std::tolower(static_cast<unsigned char>(backRank[f])));
    }
    setupFromMailbox();
    initZobrist();
    resetRepetitionTable();
}

Position::Position(const std::string& fen) {
    mailbox_.fill(' ');
    std::istringstream iss(fen);
    std::string placement, activeColor, castling, enPassant;
    int halfmoveClock = 0;
    iss >> placement >> activeColor >> castling >> enPassant;
    if (!(iss >> halfmoveClock)) halfmoveClock = 0;

    int r = 7, c = 0;
    for (char ch : placement) {
        if (ch == '/') {
            r--;
            c = 0;
        } else if (std::isdigit(static_cast<unsigned char>(ch))) {
            c += ch - '0';
        } else {
            mailbox_[squareOf(c, r)] = ch;
            c++;
        }
    }

    turn_ = (activeColor == "b") ? chess::Color::Black : chess::Color::White;

    castlingRights_ = {false, false, false, false};
    for (char ch : castling) {
        if (ch == 'K') castlingRights_[0] = true;
        else if (ch == 'Q') castlingRights_[1] = true;
        else if (ch == 'k') castlingRights_[2] = true;
        else if (ch == 'q') castlingRights_[3] = true;
    }

    if (enPassant != "-" && enPassant.size() == 2) {
        enPassantTarget_ = parseSquareUci(enPassant);
    } else {
        enPassantTarget_.reset();
    }

    halfMoves_ = halfmoveClock;

    setupFromMailbox();
    initZobrist();
    resetRepetitionTable();
}

int Position::parseSquareUci(const std::string& s) {
    int file = s[0] - 'a';
    int rank = s[1] - '1';
    return squareOf(file, rank);
}

std::string Position::squareToUci(int sq) {
    std::string s;
    s += static_cast<char>('a' + fileOf(sq));
    s += static_cast<char>('1' + rankOf(sq));
    return s;
}

chess::BoardArray Position::toBoardArray() const {
    chess::BoardArray board;
    for (auto& row : board) row.fill(' ');
    for (int sq = 0; sq < 64; sq++) {
        if (mailbox_[sq] == ' ') continue;
        int r = 7 - rankOf(sq);
        int c = fileOf(sq);
        board[r][c] = mailbox_[sq];
    }
    return board;
}

std::string Position::toFen() const {
    std::ostringstream oss;
    for (int r = 7; r >= 0; r--) {
        int empty = 0;
        for (int c = 0; c < 8; c++) {
            char p = mailbox_[squareOf(c, r)];
            if (p == ' ') {
                empty++;
            } else {
                if (empty > 0) { oss << empty; empty = 0; }
                oss << p;
            }
        }
        if (empty > 0) oss << empty;
        if (r != 0) oss << '/';
    }
    oss << ' ' << (turn_ == chess::Color::White ? 'w' : 'b') << ' ';
    std::string castling;
    if (castlingRights_[0]) castling += 'K';
    if (castlingRights_[1]) castling += 'Q';
    if (castlingRights_[2]) castling += 'k';
    if (castlingRights_[3]) castling += 'q';
    oss << (castling.empty() ? "-" : castling) << ' ';
    oss << (enPassantTarget_.has_value() ? squareToUci(*enPassantTarget_) : "-");
    oss << ' ' << halfMoves_ << " 1";
    return oss.str();
}

namespace {
int seePieceValue(char pieceLower) {
    switch (pieceLower) {
        case 'p': return 100;
        case 'n': return 320;
        case 'b': return 330;
        case 'r': return 500;
        case 'q': return 900;
        case 'k': return 20000;
        default: return 0;
    }
}
}

Bitboard Position::attackersTo(int sq, Bitboard occ) const {
    Bitboard attackers = 0;
    attackers |= knightAttacks[sq] & (pieces_[0][1] | pieces_[1][1]);
    attackers |= kingAttacks[sq] & (pieces_[0][5] | pieces_[1][5]);
    attackers |= blackPawnAttacks[sq] & pieces_[0][0];
    attackers |= whitePawnAttacks[sq] & pieces_[1][0];
    Bitboard rq = (pieces_[0][3] | pieces_[0][4] | pieces_[1][3] | pieces_[1][4]);
    attackers |= rookAttacks(sq, occ) & rq;
    Bitboard bq = (pieces_[0][2] | pieces_[0][4] | pieces_[1][2] | pieces_[1][4]);
    attackers |= bishopAttacks(sq, occ) & bq;
    return attackers & occ;
}

int Position::seeSwing(Bitboard occ, int sq, bool attackerIsWhite, int standValue) const {
    Bitboard attackers = attackersTo(sq, occ) & occupied_[attackerIsWhite ? 0 : 1] & occ;
    if (!attackers) return 0;

    int bestSq = -1, bestVal = 1'000'000;
    Bitboard a = attackers;
    while (a) {
        int s = popLsb(a);
        int v = seePieceValue(static_cast<char>(std::tolower(static_cast<unsigned char>(mailbox_[s]))));
        if (v < bestVal) {
            bestVal = v;
            bestSq = s;
        }
    }

    Bitboard newOcc = occ & ~(1ULL << bestSq);

    if (bestVal == seePieceValue('k') &&
        (attackersTo(sq, newOcc) & occupied_[attackerIsWhite ? 1 : 0] & newOcc)) {
        return 0;
    }

    int gain = standValue - seeSwing(newOcc, sq, !attackerIsWhite, bestVal);
    return std::max(0, gain);
}

int Position::see(int from, int to) const {
    char attackerPiece = mailbox_[from];
    char targetPiece = mailbox_[to];
    bool isEnPassant = targetPiece == ' ' &&
                       std::tolower(static_cast<unsigned char>(attackerPiece)) == 'p' &&
                       enPassantTarget_.has_value() && *enPassantTarget_ == to;
    int targetVal = isEnPassant ? seePieceValue('p')
                                 : seePieceValue(static_cast<char>(std::tolower(static_cast<unsigned char>(targetPiece))));
    int attackerVal = seePieceValue(static_cast<char>(std::tolower(static_cast<unsigned char>(attackerPiece))));

    Bitboard occ = allOccupied_ & ~(1ULL << from);
    bool opponentIsWhite = !isUpperPiece(attackerPiece);

    return targetVal - seeSwing(occ, to, opponentIsWhite, attackerVal);
}

bool Position::squareAttackedBy(int sq, chess::Color attacker) const {
    bool attackerWhite = attacker == chess::Color::White;
    int side = attackerWhite ? 0 : 1;

    if (knightAttacks[sq] & pieces_[side][1]) return true;
    if (kingAttacks[sq] & pieces_[side][5]) return true;

    Bitboard pawnAttackersFromSq = attackerWhite ? blackPawnAttacks[sq] : whitePawnAttacks[sq];
    if (pawnAttackersFromSq & pieces_[side][0]) return true;

    Bitboard rq = pieces_[side][3] | pieces_[side][4];
    if (rookAttacks(sq, allOccupied_) & rq) return true;
    Bitboard bq = pieces_[side][2] | pieces_[side][4];
    if (bishopAttacks(sq, allOccupied_) & bq) return true;

    return false;
}

bool Position::inCheck() const {
    int kingSq = (turn_ == chess::Color::White) ? whiteKingSq_ : blackKingSq_;
    chess::Color enemy = (turn_ == chess::Color::White) ? chess::Color::Black : chess::Color::White;
    return squareAttackedBy(kingSq, enemy);
}

BBUndo Position::makeMove(int from, int to, char promotion) {
    char piece = mailbox_[from];
    char captured = mailbox_[to];

    BBUndo undo;
    undo.from = from;
    undo.to = to;
    undo.piece = piece;
    undo.captured = captured;
    undo.prevCastlingRights = castlingRights_;
    undo.prevEnPassant = enPassantTarget_;
    undo.prevHash = zobristHash_;
    undo.prevHalfMoves = halfMoves_;
    undo.moveType = MoveType::Normal;

    zobristHash_ ^= zobristTable_[from][zobristPieceIndex(piece)];
    if (captured != ' ') {
        zobristHash_ ^= zobristTable_[to][zobristPieceIndex(captured)];
        halfMoves_ = 0;
    } else if (std::tolower(static_cast<unsigned char>(piece)) == 'p') {
        halfMoves_ = 0;
    } else {
        halfMoves_++;
    }

    clearPiece(from);
    if (captured != ' ') clearPiece(to);
    setPiece(to, piece);
    zobristHash_ ^= zobristTable_[to][zobristPieceIndex(piece)];

    if (piece == 'K') {
        whiteKingSq_ = to;
        castlingRights_[0] = false;
        castlingRights_[1] = false;
        if (to - from == 2 || from - to == 2) {
            undo.moveType = MoveType::Castle;
            if (to > from) {
                clearPiece(kWhiteRookHStart);
                setPiece(kWhiteRookHStart - 2, 'R');
                zobristHash_ ^= zobristTable_[kWhiteRookHStart][zobristPieceIndex('R')];
                zobristHash_ ^= zobristTable_[kWhiteRookHStart - 2][zobristPieceIndex('R')];
                undo.rookFrom = kWhiteRookHStart;
                undo.rookTo = kWhiteRookHStart - 2;
            } else {
                clearPiece(kWhiteRookAStart);
                setPiece(kWhiteRookAStart + 3, 'R');
                zobristHash_ ^= zobristTable_[kWhiteRookAStart][zobristPieceIndex('R')];
                zobristHash_ ^= zobristTable_[kWhiteRookAStart + 3][zobristPieceIndex('R')];
                undo.rookFrom = kWhiteRookAStart;
                undo.rookTo = kWhiteRookAStart + 3;
            }
        }
    } else if (piece == 'k') {
        blackKingSq_ = to;
        castlingRights_[2] = false;
        castlingRights_[3] = false;
        if (to - from == 2 || from - to == 2) {
            undo.moveType = MoveType::Castle;
            if (to > from) {
                clearPiece(kBlackRookHStart);
                setPiece(kBlackRookHStart - 2, 'r');
                zobristHash_ ^= zobristTable_[kBlackRookHStart][zobristPieceIndex('r')];
                zobristHash_ ^= zobristTable_[kBlackRookHStart - 2][zobristPieceIndex('r')];
                undo.rookFrom = kBlackRookHStart;
                undo.rookTo = kBlackRookHStart - 2;
            } else {
                clearPiece(kBlackRookAStart);
                setPiece(kBlackRookAStart + 3, 'r');
                zobristHash_ ^= zobristTable_[kBlackRookAStart][zobristPieceIndex('r')];
                zobristHash_ ^= zobristTable_[kBlackRookAStart + 3][zobristPieceIndex('r')];
                undo.rookFrom = kBlackRookAStart;
                undo.rookTo = kBlackRookAStart + 3;
            }
        }
    }

    if (piece == 'R') {
        if (from == kWhiteRookAStart) castlingRights_[1] = false;
        if (from == kWhiteRookHStart) castlingRights_[0] = false;
    } else if (piece == 'r') {
        if (from == kBlackRookAStart) castlingRights_[3] = false;
        if (from == kBlackRookHStart) castlingRights_[2] = false;
    }
    if (captured == 'R') {
        if (to == kWhiteRookAStart) castlingRights_[1] = false;
        if (to == kWhiteRookHStart) castlingRights_[0] = false;
    } else if (captured == 'r') {
        if (to == kBlackRookAStart) castlingRights_[3] = false;
        if (to == kBlackRookHStart) castlingRights_[2] = false;
    }

    enPassantTarget_.reset();
    if (std::tolower(static_cast<unsigned char>(piece)) == 'p') {
        int fileDelta = fileOf(to) - fileOf(from);
        int rankDelta = rankOf(to) - rankOf(from);
        if (rankDelta == 2 || rankDelta == -2) {
            int epSq = squareOf(fileOf(from), (rankOf(from) + rankOf(to)) / 2);
            bool moverIsWhite = isUpperPiece(piece);
            int enemySide = moverIsWhite ? 1 : 0;
            Bitboard enemyPawnAttackers = (moverIsWhite ? whitePawnAttacks[epSq] : blackPawnAttacks[epSq]) &
                                           pieces_[enemySide][0];
            if (enemyPawnAttackers) enPassantTarget_ = epSq;
        }
        if (fileDelta != 0 && captured == ' ') {
            undo.moveType = MoveType::EnPassant;
            int capSq = squareOf(fileOf(to), rankOf(from));
            undo.epCapturedSquare = capSq;
            undo.epCapturedPiece = mailbox_[capSq];
            zobristHash_ ^= zobristTable_[capSq][zobristPieceIndex(undo.epCapturedPiece)];
            clearPiece(capSq);
        }

        int promoRank = isUpperPiece(piece) ? 7 : 0;
        if (rankOf(to) == promoRank) {
            undo.moveType = MoveType::Promote;
            char prom = isUpperPiece(piece) ? static_cast<char>(std::toupper(static_cast<unsigned char>(promotion)))
                                             : static_cast<char>(std::tolower(static_cast<unsigned char>(promotion)));
            zobristHash_ ^= zobristTable_[to][zobristPieceIndex(piece)];
            clearPiece(to);
            setPiece(to, prom);
            zobristHash_ ^= zobristTable_[to][zobristPieceIndex(prom)];
            undo.promoteFrom = piece;
        }
    }

    for (int i = 0; i < 4; i++) {
        if (castlingRights_[i] != undo.prevCastlingRights[i]) zobristHash_ ^= zobristCastling_[i];
    }
    if (undo.prevEnPassant) zobristHash_ ^= zobristEnPassant_[fileOf(*undo.prevEnPassant)];
    if (enPassantTarget_) zobristHash_ ^= zobristEnPassant_[fileOf(*enPassantTarget_)];

    turn_ = (turn_ == chess::Color::White) ? chess::Color::Black : chess::Color::White;
    zobristHash_ ^= zobristBlackToMoveKey_;

    if (repetitionTable_.count(zobristHash_)) repetitionTable_[zobristHash_]++;
    else repetitionTable_[zobristHash_] = 1;

    return undo;
}

void Position::unmakeMove(const BBUndo& undo) {
    auto it = repetitionTable_.find(zobristHash_);
    if (it != repetitionTable_.end()) {
        it->second--;
        if (it->second == 0) repetitionTable_.erase(it);
    }

    char rookPiece = (undo.moveType == MoveType::Castle) ? mailbox_[undo.rookTo] : ' ';

    clearPiece(undo.to);
    setPiece(undo.from, undo.piece);
    if (undo.captured != ' ' && undo.moveType != MoveType::EnPassant) setPiece(undo.to, undo.captured);

    if (undo.moveType == MoveType::Castle) {
        clearPiece(undo.rookTo);
        setPiece(undo.rookFrom, rookPiece);
    } else if (undo.moveType == MoveType::EnPassant) {
        setPiece(undo.epCapturedSquare, undo.epCapturedPiece);
    }

    if (undo.piece == 'K') whiteKingSq_ = undo.from;
    else if (undo.piece == 'k') blackKingSq_ = undo.from;

    turn_ = (turn_ == chess::Color::White) ? chess::Color::Black : chess::Color::White;
    castlingRights_ = undo.prevCastlingRights;
    enPassantTarget_ = undo.prevEnPassant;
    zobristHash_ = undo.prevHash;
    halfMoves_ = undo.prevHalfMoves;
}

Position::NullUndo Position::makeNullMove() {
    NullUndo undo{enPassantTarget_, zobristHash_};
    if (enPassantTarget_) zobristHash_ ^= zobristEnPassant_[fileOf(*enPassantTarget_)];
    enPassantTarget_.reset();
    turn_ = (turn_ == chess::Color::White) ? chess::Color::Black : chess::Color::White;
    zobristHash_ ^= zobristBlackToMoveKey_;
    return undo;
}

void Position::unmakeNullMove(const NullUndo& undo) {
    turn_ = (turn_ == chess::Color::White) ? chess::Color::Black : chess::Color::White;
    enPassantTarget_ = undo.prevEnPassant;
    zobristHash_ = undo.prevHash;
}

bool Position::isRepetitionDraw() const {
    auto it = repetitionTable_.find(zobristHash_);
    return it != repetitionTable_.end() && it->second >= 3;
}

bool Position::isInsufficientMaterial() const {
    int minorCount = 0;
    for (int sq = 0; sq < 64; sq++) {
        char p = mailbox_[sq];
        if (p == ' ') continue;
        char pl = static_cast<char>(std::tolower(static_cast<unsigned char>(p)));
        if (pl == 'k') continue;
        if (pl != 'n' && pl != 'b') return false;
        minorCount++;
        if (minorCount > 1) return false;
    }
    return true;
}

namespace {

struct PseudoMoveGen {
    const Position* pos;
    std::vector<BBMove> moves;
};

}

std::vector<BBMove> Position::getValidMoves() {
    std::vector<BBMove> valid;
    valid.reserve(48);

    bool white = turn_ == chess::Color::White;
    int side = white ? 0 : 1;
    int enemySide = white ? 1 : 0;
    chess::Color enemyColor = white ? chess::Color::Black : chess::Color::White;
    Bitboard ownOcc = occupied_[side];
    Bitboard enemyOcc = occupied_[enemySide];

    int kingSq = white ? whiteKingSq_ : blackKingSq_;

    int checkerCount = 0;
    Bitboard resolutionSquares = 0;
    Bitboard pinned = 0;
    std::array<Bitboard, 64> pinRay{};

    {
        Bitboard knightCheckers = knightAttacks[kingSq] & pieces_[enemySide][1];
        if (knightCheckers) {
            checkerCount += popcount(knightCheckers);
            resolutionSquares |= knightCheckers;
        }
        Bitboard pawnCheckers = (white ? whitePawnAttacks[kingSq] : blackPawnAttacks[kingSq]) & pieces_[enemySide][0];
        if (pawnCheckers) {
            checkerCount += popcount(pawnCheckers);
            resolutionSquares |= pawnCheckers;
        }

        Bitboard snipers = (rookAttacks(kingSq, 0) & (pieces_[enemySide][3] | pieces_[enemySide][4])) |
                            (bishopAttacks(kingSq, 0) & (pieces_[enemySide][2] | pieces_[enemySide][4]));
        Bitboard sn = snipers;
        while (sn) {
            int sniperSq = popLsb(sn);
            Bitboard between = squaresBetween[kingSq][sniperSq] & allOccupied_;
            int c = popcount(between);
            if (c == 0) {
                checkerCount++;
                resolutionSquares |= squaresBetween[kingSq][sniperSq];
                resolutionSquares |= (1ULL << sniperSq);
            } else if (c == 1) {
                int blockerSq = bitscanForward(between);
                if (ownOcc & (1ULL << blockerSq)) {
                    pinned |= (1ULL << blockerSq);
                    pinRay[blockerSq] = squaresBetween[kingSq][sniperSq] | (1ULL << sniperSq);
                }
            }
        }
    }

    auto addPromotions = [&](int from, int to) {
        for (char p : {'q', 'r', 'b', 'n'}) valid.push_back({from, to, p});
    };

    auto legalNonKingDestFilter = [&](int from, Bitboard destinations) {
        if (checkerCount >= 2) return static_cast<Bitboard>(0);
        if (checkerCount == 1) destinations &= resolutionSquares;
        if (pinned & (1ULL << from)) destinations &= pinRay[from];
        return destinations;
    };

    if (checkerCount < 2) {
        Bitboard pawns = pieces_[side][0];
        Bitboard p = pawns;
        while (p) {
            int from = popLsb(p);
            int dir = white ? 8 : -8;
            int startRank = white ? 1 : 6;
            int promoRank = white ? 7 : 0;
            int to1 = from + dir;

            Bitboard pushDest = 0;
            if (onBoardFR(fileOf(from), rankOf(from) + (white ? 1 : -1)) && mailbox_[to1] == ' ') {
                pushDest |= (1ULL << to1);
                if (rankOf(from) == startRank) {
                    int to2 = from + 2 * dir;
                    if (mailbox_[to2] == ' ') pushDest |= (1ULL << to2);
                }
            }
            pushDest = legalNonKingDestFilter(from, pushDest);
            while (pushDest) {
                int to = popLsb(pushDest);
                if (rankOf(to) == promoRank) addPromotions(from, to);
                else valid.push_back({from, to});
            }

            Bitboard attackSquares = (white ? whitePawnAttacks[from] : blackPawnAttacks[from]);
            Bitboard captureDest = attackSquares & enemyOcc;
            Bitboard epDest = 0;
            if (enPassantTarget_.has_value() && (attackSquares & (1ULL << *enPassantTarget_))) {
                epDest |= (1ULL << *enPassantTarget_);
            }
            captureDest = legalNonKingDestFilter(from, captureDest);
            while (captureDest) {
                int to = popLsb(captureDest);
                if (rankOf(to) == promoRank) addPromotions(from, to);
                else valid.push_back({from, to});
            }
            while (epDest) {
                int to = popLsb(epDest);
                valid.push_back({from, to});
            }
        }

        // Knights
        Bitboard knights = pieces_[side][1];
        Bitboard n = knights;
        while (n) {
            int from = popLsb(n);
            Bitboard dest = legalNonKingDestFilter(from, knightAttacks[from] & ~ownOcc);
            while (dest) valid.push_back({from, popLsb(dest)});
        }

        // Bishops/Rooks/Queens
        for (int pt : {2, 3, 4}) {
            Bitboard sliders = pieces_[side][pt];
            Bitboard s = sliders;
            while (s) {
                int from = popLsb(s);
                Bitboard atk = 0;
                if (pt == 2) atk = bishopAttacks(from, allOccupied_);
                else if (pt == 3) atk = rookAttacks(from, allOccupied_);
                else atk = queenAttacks(from, allOccupied_);
                Bitboard dest = legalNonKingDestFilter(from, atk & ~ownOcc);
                while (dest) valid.push_back({from, popLsb(dest)});
            }
        }
    }

    // King moves (including castling) — always computed via the slow make/unmake path.
    {
        Bitboard dest = kingAttacks[kingSq] & ~ownOcc;
        Bitboard d = dest;
        while (d) {
            int to = popLsb(d);
            valid.push_back({kingSq, to});
        }

        if (checkerCount == 0) {
            if (white) {
                if (castlingRights_[0] && mailbox_[5] == ' ' && mailbox_[6] == ' ' &&
                    !squareAttackedBy(5, chess::Color::Black) && !squareAttackedBy(6, chess::Color::Black)) {
                    valid.push_back({4, 6});
                }
                if (castlingRights_[1] && mailbox_[3] == ' ' && mailbox_[2] == ' ' && mailbox_[1] == ' ' &&
                    !squareAttackedBy(3, chess::Color::Black) && !squareAttackedBy(2, chess::Color::Black)) {
                    valid.push_back({4, 2});
                }
            } else {
                if (castlingRights_[2] && mailbox_[61] == ' ' && mailbox_[62] == ' ' &&
                    !squareAttackedBy(61, chess::Color::White) && !squareAttackedBy(62, chess::Color::White)) {
                    valid.push_back({60, 62});
                }
                if (castlingRights_[3] && mailbox_[59] == ' ' && mailbox_[58] == ' ' && mailbox_[57] == ' ' &&
                    !squareAttackedBy(59, chess::Color::White) && !squareAttackedBy(58, chess::Color::White)) {
                    valid.push_back({60, 58});
                }
            }
        }
    }

    // Final legality pass for king moves and en passant captures: verify via make/unmake,
    // exactly like the array engine's hybrid approach — these are the cases the O(1)
    // pin/checker bitboards above don't (king moving away along a check ray) or can't
    // (en-passant discovered check) safely cover.
    std::vector<BBMove> finalMoves;
    finalMoves.reserve(valid.size());
    for (const auto& m : valid) {
        char movingPiece = mailbox_[m.from];
        bool isKingMove = std::tolower(static_cast<unsigned char>(movingPiece)) == 'k';
        bool isEnPassant = std::tolower(static_cast<unsigned char>(movingPiece)) == 'p' &&
                            fileOf(m.from) != fileOf(m.to) && mailbox_[m.to] == ' ';
        if (isKingMove || isEnPassant) {
            char promo = (m.promotion == ' ') ? 'q' : m.promotion;
            BBUndo undo = makeMove(m.from, m.to, promo);
            bool legal = !squareAttackedBy((turn_ == chess::Color::White) ? blackKingSq_ : whiteKingSq_,
                                            (turn_ == chess::Color::White) ? chess::Color::White : chess::Color::Black);
            unmakeMove(undo);
            if (legal) finalMoves.push_back(m);
        } else {
            finalMoves.push_back(m);
        }
    }

    (void)enemyColor;
    return finalMoves;
}

}
