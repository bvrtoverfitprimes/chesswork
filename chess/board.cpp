#include "board.h"

#include <cctype>
#include <iostream>
#include <sstream>
#include <random>

namespace chess {

namespace {

std::mt19937_64 rng(std::random_device{}());

bool isUpperPiece(char p) { return p != ' ' && std::isupper(static_cast<unsigned char>(p)); }

}

int Game::pieceIndex(char piece) {
    switch (piece) {
        case 'P': return 1;
        case 'N': return 2;
        case 'B': return 3;
        case 'R': return 4;
        case 'Q': return 5;
        case 'K': return 6;
        case 'p': return 7;
        case 'n': return 8;
        case 'b': return 9;
        case 'r': return 10;
        case 'q': return 11;
        case 'k': return 12;
        default: return 0;
    }
}

Game::Game() {
    for (auto& row : board_) row.fill(' ');
    board_[6].fill('P');
    board_[7] = {'R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R'};
    board_[1].fill('p');
    board_[0] = {'r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'};

    findKings();
    initZobrist();
    resetRepetitionTable();
}

Game::Game(const std::string& fen) {
    for (auto& row : board_) row.fill(' ');

    std::istringstream iss(fen);
    std::string placement, activeColor, castling, enPassant;
    int halfmoveClock = 0;
    iss >> placement >> activeColor >> castling >> enPassant;
    if (!(iss >> halfmoveClock)) halfmoveClock = 0;

    int r = 0, c = 0;
    for (char ch : placement) {
        if (ch == '/') {
            r++;
            c = 0;
        } else if (std::isdigit(static_cast<unsigned char>(ch))) {
            c += ch - '0';
        } else {
            board_[r][c] = ch;
            c++;
        }
    }

    turn_ = (activeColor == "b") ? Color::Black : Color::White;

    castlingRights_ = {false, false, false, false};
    for (char ch : castling) {
        if (ch == 'K') castlingRights_[0] = true;
        else if (ch == 'Q') castlingRights_[1] = true;
        else if (ch == 'k') castlingRights_[2] = true;
        else if (ch == 'q') castlingRights_[3] = true;
    }

    if (enPassant != "-" && enPassant.size() == 2) {
        enPassantTarget_ = parseSquare(enPassant);
    } else {
        enPassantTarget_.reset();
    }

    halfMoves_ = halfmoveClock;

    findKings();
    initZobrist();
    resetRepetitionTable();
}

void Game::findKings() {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board_[r][c] == 'K') whiteKing_ = {r, c};
            else if (board_[r][c] == 'k') blackKing_ = {r, c};
        }
    }
}

bool Game::inCheckColor(Color c) const {
    Pos king = (c == Color::White) ? whiteKing_ : blackKing_;
    return isAttacked(board_, king.r, king.c, c);
}

void Game::resetRepetitionTable() {
    repetitionTable_.clear();
    repetitionTable_[zobristHash_] = 1;
}

void Game::initZobrist() {
    for (auto& row : zobristTable_)
        for (auto& v : row)
            v = rng();
    zobristBlackToMoveKey_ = rng();

    uint64_t h = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board_[r][c];
            if (p != ' ') {
                h ^= zobristTable_[r * 8 + c][pieceIndex(p)];
            }
        }
    }
    if (turn_ == Color::Black) h ^= zobristBlackToMoveKey_;
    zobristHash_ = h;
}

void Game::printBoard() const {
    for (int i = 0; i < 8; i++) {
        std::cout << "[" << (8 - i) << "] [";
        for (int c = 0; c < 8; c++) {
            std::cout << "'" << board_[i][c] << "'";
            if (c != 7) std::cout << ", ";
        }
        std::cout << "]\n\n";
    }
    std::cout << "    ";
    for (char letter : LATERAL) std::cout << "['" << letter << "']";
    std::cout << "\n";
}

Pos Game::parseSquare(const std::string& s) {
    int file = s[0] - 'a';
    int rank = s[1] - '0';
    return {8 - rank, file};
}

std::string Game::toFen() const {
    std::ostringstream oss;
    for (int r = 0; r < 8; r++) {
        int empty = 0;
        for (int c = 0; c < 8; c++) {
            char p = board_[r][c];
            if (p == ' ') {
                empty++;
            } else {
                if (empty > 0) {
                    oss << empty;
                    empty = 0;
                }
                oss << p;
            }
        }
        if (empty > 0) oss << empty;
        if (r != 7) oss << '/';
    }

    oss << ' ' << (turn_ == Color::White ? 'w' : 'b') << ' ';

    std::string castling;
    if (castlingRights_[0]) castling += 'K';
    if (castlingRights_[1]) castling += 'Q';
    if (castlingRights_[2]) castling += 'k';
    if (castlingRights_[3]) castling += 'q';
    oss << (castling.empty() ? "-" : castling) << ' ';

    oss << (enPassantTarget_.has_value() ? squareToStr(*enPassantTarget_) : "-");
    oss << ' ' << halfMoves_ << " 1";

    return oss.str();
}

std::string Game::squareToStr(Pos p) {
    std::string s;
    s += LATERAL[p.c];
    s += std::to_string(8 - p.r);
    return s;
}

UndoMove Game::makeMove(Pos from, Pos to, char promotion) {
    int r1 = from.r, c1 = from.c;
    int r2 = to.r, c2 = to.c;
    char piece = board_[r1][c1];
    char captured = board_[r2][c2];

    UndoMove undo;
    undo.from = from;
    undo.to = to;
    undo.piece = piece;
    undo.captured = captured;
    undo.prevCastlingRights = castlingRights_;
    undo.prevEnPassant = enPassantTarget_;
    undo.prevHash = zobristHash_;
    undo.prevHalfMoves = halfMoves_;
    undo.moveType = MoveType::Normal;

    zobristHash_ ^= zobristTable_[r1 * 8 + c1][pieceIndex(piece)];
    if (captured != ' ') {
        zobristHash_ ^= zobristTable_[r2 * 8 + c2][pieceIndex(captured)];
        halfMoves_ = 0;
    } else if (std::tolower(static_cast<unsigned char>(piece)) == 'p') {
        halfMoves_ = 0;
    } else {
        halfMoves_++;
    }

    board_[r1][c1] = ' ';
    board_[r2][c2] = piece;
    zobristHash_ ^= zobristTable_[r2 * 8 + c2][pieceIndex(piece)];

    if (piece == 'K') {
        whiteKing_ = to;
        castlingRights_[0] = false;
        castlingRights_[1] = false;
        if (std::abs(c2 - c1) == 2) {
            undo.moveType = MoveType::Castle;
            if (c2 > c1) {
                board_[7][7] = ' ';
                board_[7][5] = 'R';
                zobristHash_ ^= zobristTable_[7 * 8 + 7][pieceIndex('R')];
                zobristHash_ ^= zobristTable_[7 * 8 + 5][pieceIndex('R')];
                undo.rookFrom = {7, 7};
                undo.rookTo = {7, 5};
            } else {
                board_[7][0] = ' ';
                board_[7][3] = 'R';
                zobristHash_ ^= zobristTable_[7 * 8 + 0][pieceIndex('R')];
                zobristHash_ ^= zobristTable_[7 * 8 + 3][pieceIndex('R')];
                undo.rookFrom = {7, 0};
                undo.rookTo = {7, 3};
            }
        }
    } else if (piece == 'k') {
        blackKing_ = to;
        castlingRights_[2] = false;
        castlingRights_[3] = false;
        if (std::abs(c2 - c1) == 2) {
            undo.moveType = MoveType::Castle;
            if (c2 > c1) {
                board_[0][7] = ' ';
                board_[0][5] = 'r';
                zobristHash_ ^= zobristTable_[0 * 8 + 7][pieceIndex('r')];
                zobristHash_ ^= zobristTable_[0 * 8 + 5][pieceIndex('r')];
                undo.rookFrom = {0, 7};
                undo.rookTo = {0, 5};
            } else {
                board_[0][0] = ' ';
                board_[0][3] = 'r';
                zobristHash_ ^= zobristTable_[0 * 8 + 0][pieceIndex('r')];
                zobristHash_ ^= zobristTable_[0 * 8 + 3][pieceIndex('r')];
                undo.rookFrom = {0, 0};
                undo.rookTo = {0, 3};
            }
        }
    }

    if (piece == 'R') {
        if (r1 == 7 && c1 == 0) castlingRights_[1] = false;
        if (r1 == 7 && c1 == 7) castlingRights_[0] = false;
    } else if (piece == 'r') {
        if (r1 == 0 && c1 == 0) castlingRights_[3] = false;
        if (r1 == 0 && c1 == 7) castlingRights_[2] = false;
    }

    if (captured == 'R') {
        if (r2 == 7 && c2 == 0) castlingRights_[1] = false;
        if (r2 == 7 && c2 == 7) castlingRights_[0] = false;
    } else if (captured == 'r') {
        if (r2 == 0 && c2 == 0) castlingRights_[3] = false;
        if (r2 == 0 && c2 == 7) castlingRights_[2] = false;
    }

    enPassantTarget_.reset();
    if (std::tolower(static_cast<unsigned char>(piece)) == 'p') {
        if (std::abs(r2 - r1) == 2) {
            enPassantTarget_ = Pos{(r1 + r2) / 2, c1};
        }
        if (c1 != c2 && captured == ' ') {
            undo.moveType = MoveType::EnPassant;
            int capR = r1, capC = c2;
            undo.epCapturedSquare = {capR, capC};
            undo.epCapturedPiece = board_[capR][capC];
            zobristHash_ ^= zobristTable_[capR * 8 + capC][pieceIndex(undo.epCapturedPiece)];
            board_[capR][capC] = ' ';
        }

        if (r2 == 0 || r2 == 7) {
            undo.moveType = MoveType::Promote;
            char prom = isUpperPiece(piece) ? std::toupper(static_cast<unsigned char>(promotion))
                                             : std::tolower(static_cast<unsigned char>(promotion));
            zobristHash_ ^= zobristTable_[r2 * 8 + c2][pieceIndex(piece)];
            board_[r2][c2] = prom;
            zobristHash_ ^= zobristTable_[r2 * 8 + c2][pieceIndex(prom)];
            undo.promoteFrom = piece;
        }
    }

    turn_ = (turn_ == Color::White) ? Color::Black : Color::White;
    zobristHash_ ^= zobristBlackToMoveKey_;

    if (repetitionTable_.count(zobristHash_)) repetitionTable_[zobristHash_]++;
    else repetitionTable_[zobristHash_] = 1;

    return undo;
}

void Game::unmakeMove(const UndoMove& undo) {
    int r1 = undo.from.r, c1 = undo.from.c;
    int r2 = undo.to.r, c2 = undo.to.c;

    auto it = repetitionTable_.find(zobristHash_);
    if (it != repetitionTable_.end()) {
        it->second--;
        if (it->second == 0) repetitionTable_.erase(it);
    }

    board_[r1][c1] = undo.piece;
    board_[r2][c2] = undo.captured;

    if (undo.piece == 'K') whiteKing_ = undo.from;
    else if (undo.piece == 'k') blackKing_ = undo.from;

    if (undo.moveType == MoveType::Castle) {
        board_[undo.rookFrom.r][undo.rookFrom.c] = board_[undo.rookTo.r][undo.rookTo.c];
        board_[undo.rookTo.r][undo.rookTo.c] = ' ';
    } else if (undo.moveType == MoveType::EnPassant) {
        board_[r1][c2] = undo.epCapturedPiece;
    } else if (undo.moveType == MoveType::Promote) {
        board_[r1][c1] = undo.promoteFrom;
        board_[r2][c2] = undo.captured;
    }

    turn_ = (turn_ == Color::White) ? Color::Black : Color::White;
    castlingRights_ = undo.prevCastlingRights;
    enPassantTarget_ = undo.prevEnPassant;
    zobristHash_ = undo.prevHash;
    halfMoves_ = undo.prevHalfMoves;
}

Game::NullUndo Game::makeNullMove() {
    NullUndo undo{enPassantTarget_, zobristHash_};
    enPassantTarget_.reset();
    turn_ = (turn_ == Color::White) ? Color::Black : Color::White;
    zobristHash_ ^= zobristBlackToMoveKey_;
    return undo;
}

void Game::unmakeNullMove(const NullUndo& undo) {
    turn_ = (turn_ == Color::White) ? Color::Black : Color::White;
    enPassantTarget_ = undo.prevEnPassant;
    zobristHash_ = undo.prevHash;
}

std::vector<Move> Game::getValidMoves() {
    std::vector<Move> valid;
    auto moves = genPseudoMoves(board_, turn_, castlingRights_, enPassantTarget_);
    Pos ownKing = (turn_ == Color::White) ? whiteKing_ : blackKing_;
    LegalMoveContext ctx = computeLegalMoveContext(board_, turn_, ownKing);

    for (const auto& m : moves) {
        char promo = (m.promotion == ' ') ? 'q' : m.promotion;
        char movingPiece = board_[m.from.r][m.from.c];
        bool isKingMove = std::tolower(static_cast<unsigned char>(movingPiece)) == 'k';
        bool isEnPassantCapture = std::tolower(static_cast<unsigned char>(movingPiece)) == 'p' &&
                                   m.from.c != m.to.c && board_[m.to.r][m.to.c] == ' ';

        bool legal;
        if (isKingMove || isEnPassantCapture) {
            UndoMove undo = makeMove(m.from, m.to, promo);
            Color justMoved = (turn_ == Color::White) ? Color::Black : Color::White;
            legal = !inCheckColor(justMoved);
            unmakeMove(undo);
        } else {
            legal = isLegalFast(ctx, m);
        }

        if (legal) valid.push_back(m);
    }
    return valid;
}

std::vector<std::string> Game::getValidMovesUci() {
    std::vector<std::string> valid;
    auto moves = getValidMoves();
    valid.reserve(moves.size());
    for (const auto& m : moves) {
        std::string uci = squareToStr(m.from) + squareToStr(m.to);
        if (m.promotion != ' ') uci += m.promotion;
        valid.push_back(std::move(uci));
    }
    return valid;
}

std::vector<std::string> Game::getValidMovesUciSlow() {
    std::vector<std::string> valid;
    auto moves = genPseudoMoves(board_, turn_, castlingRights_, enPassantTarget_);
    for (const auto& m : moves) {
        char promo = (m.promotion == ' ') ? 'q' : m.promotion;
        UndoMove undo = makeMove(m.from, m.to, promo);
        Color justMoved = (turn_ == Color::White) ? Color::Black : Color::White;
        if (!isCheck(board_, justMoved)) {
            std::string uci = squareToStr(m.from) + squareToStr(m.to);
            if (m.promotion != ' ') uci += m.promotion;
            valid.push_back(uci);
        }
        unmakeMove(undo);
    }
    return valid;
}

bool Game::inCheck() const {
    return inCheckColor(turn_);
}

bool Game::isCheckmate() {
    return inCheck() && getValidMovesUci().empty();
}

bool Game::isStalemate() {
    return !inCheck() && getValidMovesUci().empty();
}

bool Game::isRepetitionDraw() const {
    auto it = repetitionTable_.find(zobristHash_);
    return it != repetitionTable_.end() && it->second >= 3;
}

bool Game::isFiftyMoveDraw() const {
    return halfMoves_ >= 100;
}

bool Game::isInsufficientMaterial() const {
    int minorCount = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board_[r][c];
            if (p == ' ') continue;
            char pl = std::tolower(static_cast<unsigned char>(p));
            if (pl == 'k') continue;
            if (pl != 'n' && pl != 'b') return false;
            minorCount++;
            if (minorCount > 1) return false;
        }
    }
    return true;
}

void Game::run() {
    printBoard();
    while (true) {
        if (isFiftyMoveDraw()) {
            std::cout << "Draw by fifty-move rule.\n";
            break;
        }
        if (isRepetitionDraw()) {
            std::cout << "Draw by repetition.\n";
            break;
        }
        if (isInsufficientMaterial()) {
            std::cout << "Draw by insufficient material.\n";
            break;
        }

        auto legal = getValidMovesUci();
        if (legal.empty()) {
            if (inCheck()) {
                std::cout << "Checkmate! " << (turn_ == Color::White ? "Black" : "White") << " wins.\n";
            } else {
                std::cout << "Stalemate!\n";
            }
            break;
        }

        std::cout << "\n" << (turn_ == Color::White ? "white" : "black") << "'s move: ";
        std::string moveInput;
        if (!(std::cin >> moveInput)) break;

        bool found = false;
        for (const auto& lm : legal) {
            if (lm == moveInput) { found = true; break; }
        }

        if (found) {
            Pos from = parseSquare(moveInput.substr(0, 2));
            Pos to = parseSquare(moveInput.substr(2, 2));
            char promo = (moveInput.size() == 5) ? moveInput[4] : 'q';
            makeMove(from, to, promo);
            printBoard();
        } else {
            std::cout << "Invalid move.\n";
        }
    }
}

}
