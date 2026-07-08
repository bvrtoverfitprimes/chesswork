#include "pieces.h"

#include <cctype>

namespace chess {

const std::array<char, 8> LATERAL = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'};

namespace {

const std::array<std::pair<int, int>, 8> KNIGHT_OFFSETS = {{
    {2, 1}, {2, -1}, {-2, 1}, {-2, -1}, {1, 2}, {1, -2}, {-1, 2}, {-1, -2}
}};

const std::array<std::pair<int, int>, 8> KING_OFFSETS = {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
}};

const std::array<std::pair<int, int>, 4> BISHOP_DIRS = {{
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
}};

const std::array<std::pair<int, int>, 4> ROOK_DIRS = {{
    {1, 0}, {-1, 0}, {0, 1}, {0, -1}
}};

const std::array<std::pair<int, int>, 8> QUEEN_DIRS = {{
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}
}};

bool isWhitePiece(char p) { return p != ' ' && std::isupper(static_cast<unsigned char>(p)); }
bool isBlackPiece(char p) { return p != ' ' && std::islower(static_cast<unsigned char>(p)); }

bool isEnemy(char target, bool attackerIsWhite) {
    if (target == ' ') return false;
    return attackerIsWhite ? isBlackPiece(target) : isWhitePiece(target);
}

std::vector<std::pair<int, int>> dirsFor(char pieceLower) {
    if (pieceLower == 'b') return {BISHOP_DIRS.begin(), BISHOP_DIRS.end()};
    if (pieceLower == 'r') return {ROOK_DIRS.begin(), ROOK_DIRS.end()};
    return {QUEEN_DIRS.begin(), QUEEN_DIRS.end()};
}

}

bool onBoard(int r, int c) { return r >= 0 && r < 8 && c >= 0 && c < 8; }

bool isAttacked(const BoardArray& board, int r, int c, Color color) {
    bool defenderIsWhite = color == Color::White;
    auto isEnemyPiece = [&](char p) {
        if (p == ' ') return false;
        return defenderIsWhite ? isBlackPiece(p) : isWhitePiece(p);
    };

    for (auto [dr, dc] : KNIGHT_OFFSETS) {
        int tr = r + dr, tc = c + dc;
        if (onBoard(tr, tc)) {
            char p = board[tr][tc];
            if (isEnemyPiece(p) && std::tolower(static_cast<unsigned char>(p)) == 'n') return true;
        }
    }

    for (auto [dr, dc] : KING_OFFSETS) {
        int tr = r + dr, tc = c + dc;
        if (onBoard(tr, tc)) {
            char p = board[tr][tc];
            if (isEnemyPiece(p) && std::tolower(static_cast<unsigned char>(p)) == 'k') return true;
        }
    }

    for (char pType : {'b', 'r', 'q'}) {
        auto dirs = dirsFor(pType);
        for (auto [dr, dc] : dirs) {
            int tr = r + dr, tc = c + dc;
            while (onBoard(tr, tc)) {
                char p = board[tr][tc];
                if (p != ' ') {
                    char pl = std::tolower(static_cast<unsigned char>(p));
                    if (isEnemyPiece(p) && (pl == pType || pl == 'q')) return true;
                    break;
                }
                tr += dr;
                tc += dc;
            }
        }
    }

    if (defenderIsWhite) {
        if (onBoard(r - 1, c - 1) && board[r - 1][c - 1] == 'p') return true;
        if (onBoard(r - 1, c + 1) && board[r - 1][c + 1] == 'p') return true;
    } else {
        if (onBoard(r + 1, c - 1) && board[r + 1][c - 1] == 'P') return true;
        if (onBoard(r + 1, c + 1) && board[r + 1][c + 1] == 'P') return true;
    }

    return false;
}

bool isCheck(const BoardArray& board, Color color) {
    char kChar = (color == Color::White) ? 'K' : 'k';
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] == kChar) {
                return isAttacked(board, r, c, color);
            }
        }
    }
    return false;
}

LegalMoveContext computeLegalMoveContext(const BoardArray& board, Color color) {
    bool isWhite = color == Color::White;
    char kChar = isWhite ? 'K' : 'k';
    Pos king{-1, -1};
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] == kChar) king = {r, c};
        }
    }
    return computeLegalMoveContext(board, color, king);
}

LegalMoveContext computeLegalMoveContext(const BoardArray& board, Color color, Pos king) {
    LegalMoveContext ctx;
    ctx.king = king;
    bool isWhite = color == Color::White;
    int kr = ctx.king.r, kc = ctx.king.c;

    for (auto [dr, dc] : KNIGHT_OFFSETS) {
        int tr = kr + dr, tc = kc + dc;
        if (onBoard(tr, tc)) {
            char p = board[tr][tc];
            if (isEnemy(p, isWhite) && std::tolower(static_cast<unsigned char>(p)) == 'n') {
                ctx.checkerCount++;
                ctx.resolutionSquares.push_back({tr, tc});
            }
        }
    }

    if (isWhite) {
        if (onBoard(kr - 1, kc - 1) && board[kr - 1][kc - 1] == 'p') {
            ctx.checkerCount++;
            ctx.resolutionSquares.push_back({kr - 1, kc - 1});
        }
        if (onBoard(kr - 1, kc + 1) && board[kr - 1][kc + 1] == 'p') {
            ctx.checkerCount++;
            ctx.resolutionSquares.push_back({kr - 1, kc + 1});
        }
    } else {
        if (onBoard(kr + 1, kc - 1) && board[kr + 1][kc - 1] == 'P') {
            ctx.checkerCount++;
            ctx.resolutionSquares.push_back({kr + 1, kc - 1});
        }
        if (onBoard(kr + 1, kc + 1) && board[kr + 1][kc + 1] == 'P') {
            ctx.checkerCount++;
            ctx.resolutionSquares.push_back({kr + 1, kc + 1});
        }
    }

    for (auto [dr, dc] : QUEEN_DIRS) {
        bool diagonal = (dr != 0 && dc != 0);
        std::vector<Pos> emptySquares;
        Pos ownPieceSquare{-1, -1};
        bool foundOwn = false;
        int tr = kr + dr, tc = kc + dc;
        while (onBoard(tr, tc)) {
            char p = board[tr][tc];
            if (p == ' ') {
                emptySquares.push_back({tr, tc});
                tr += dr;
                tc += dc;
                continue;
            }
            bool pieceIsWhite = isWhitePiece(p);
            bool isOwnPiece = (pieceIsWhite == isWhite);
            if (isOwnPiece) {
                if (foundOwn) break;
                foundOwn = true;
                ownPieceSquare = {tr, tc};
                tr += dr;
                tc += dc;
                continue;
            }
            char pl = std::tolower(static_cast<unsigned char>(p));
            bool attacksThisDir = diagonal ? (pl == 'b' || pl == 'q') : (pl == 'r' || pl == 'q');
            if (attacksThisDir) {
                if (!foundOwn) {
                    ctx.checkerCount++;
                    for (const auto& sq : emptySquares) ctx.resolutionSquares.push_back(sq);
                    ctx.resolutionSquares.push_back({tr, tc});
                } else {
                    ctx.pinned[ownPieceSquare.r][ownPieceSquare.c] = true;
                    ctx.pinDir[ownPieceSquare.r][ownPieceSquare.c] = {dr, dc};
                }
            }
            break;
        }
    }

    return ctx;
}

bool isLegalFast(const LegalMoveContext& ctx, const Move& m) {
    if (ctx.checkerCount >= 2) return false;

    if (ctx.checkerCount == 1) {
        bool resolves = false;
        for (const auto& sq : ctx.resolutionSquares) {
            if (sq == m.to) {
                resolves = true;
                break;
            }
        }
        if (!resolves) return false;
    }

    if (ctx.pinned[m.from.r][m.from.c]) {
        auto [dr, dc] = ctx.pinDir[m.from.r][m.from.c];
        int mdr = m.to.r - m.from.r, mdc = m.to.c - m.from.c;
        if (mdr * dc - mdc * dr != 0) return false;
    }

    return true;
}

std::vector<Move> genPseudoMoves(const BoardArray& board, Color color,
                                  const CastlingRights& castlingRights,
                                  const std::optional<Pos>& enPassantTarget) {
    std::vector<Move> moves;
    moves.reserve(48);
    bool isWhite = color == Color::White;

    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;
            if (isWhite && !isWhitePiece(p)) continue;
            if (!isWhite && !isBlackPiece(p)) continue;

            char pl = std::tolower(static_cast<unsigned char>(p));

            if (pl == 'p') {
                int direction = isWhite ? -1 : 1;
                int startRow = isWhite ? 6 : 1;

                if (onBoard(r + direction, c) && board[r + direction][c] == ' ') {
                    int tr = r + direction;
                    if (tr == 0 || tr == 7) {
                        for (char promo : {'q', 'r', 'b', 'n'}) {
                            moves.push_back({{r, c}, {tr, c}, promo});
                        }
                    } else {
                        moves.push_back({{r, c}, {tr, c}});
                        if (r == startRow && board[r + 2 * direction][c] == ' ') {
                            moves.push_back({{r, c}, {r + 2 * direction, c}});
                        }
                    }
                }

                for (int dc : {-1, 1}) {
                    int tr = r + direction, tc = c + dc;
                    if (onBoard(tr, tc)) {
                        char target = board[tr][tc];
                        if (isEnemy(target, isWhite)) {
                            if (tr == 0 || tr == 7) {
                                for (char promo : {'q', 'r', 'b', 'n'}) {
                                    moves.push_back({{r, c}, {tr, tc}, promo});
                                }
                            } else {
                                moves.push_back({{r, c}, {tr, tc}});
                            }
                        } else if (enPassantTarget.has_value() &&
                                   enPassantTarget->r == tr && enPassantTarget->c == tc) {
                            moves.push_back({{r, c}, {tr, tc}});
                        }
                    }
                }
            } else if (pl == 'n') {
                for (auto [dr, dc] : KNIGHT_OFFSETS) {
                    int tr = r + dr, tc = c + dc;
                    if (onBoard(tr, tc)) {
                        char target = board[tr][tc];
                        if (target == ' ' || isEnemy(target, isWhite)) {
                            moves.push_back({{r, c}, {tr, tc}});
                        }
                    }
                }
            } else if (pl == 'k') {
                for (auto [dr, dc] : KING_OFFSETS) {
                    int tr = r + dr, tc = c + dc;
                    if (onBoard(tr, tc)) {
                        char target = board[tr][tc];
                        if (target == ' ' || isEnemy(target, isWhite)) {
                            moves.push_back({{r, c}, {tr, tc}});
                        }
                    }
                }

                if (!isCheck(board, color)) {
                    if (isWhite) {
                        if (castlingRights[0] && board[7][5] == ' ' && board[7][6] == ' ' &&
                            !isAttacked(board, 7, 5, color) && !isAttacked(board, 7, 6, color)) {
                            moves.push_back({{7, 4}, {7, 6}});
                        }
                        if (castlingRights[1] && board[7][3] == ' ' && board[7][2] == ' ' && board[7][1] == ' ' &&
                            !isAttacked(board, 7, 3, color) && !isAttacked(board, 7, 2, color)) {
                            moves.push_back({{7, 4}, {7, 2}});
                        }
                    } else {
                        if (castlingRights[2] && board[0][5] == ' ' && board[0][6] == ' ' &&
                            !isAttacked(board, 0, 5, color) && !isAttacked(board, 0, 6, color)) {
                            moves.push_back({{0, 4}, {0, 6}});
                        }
                        if (castlingRights[3] && board[0][3] == ' ' && board[0][2] == ' ' && board[0][1] == ' ' &&
                            !isAttacked(board, 0, 3, color) && !isAttacked(board, 0, 2, color)) {
                            moves.push_back({{0, 4}, {0, 2}});
                        }
                    }
                }
            } else {
                auto dirs = dirsFor(pl);
                for (auto [dr, dc] : dirs) {
                    int tr = r + dr, tc = c + dc;
                    while (onBoard(tr, tc)) {
                        char target = board[tr][tc];
                        if (target == ' ') {
                            moves.push_back({{r, c}, {tr, tc}});
                        } else {
                            if (isEnemy(target, isWhite)) {
                                moves.push_back({{r, c}, {tr, tc}});
                            }
                            break;
                        }
                        tr += dr;
                        tc += dc;
                    }
                }
            }
        }
    }
    return moves;
}

}
