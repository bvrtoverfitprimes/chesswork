#include "nnue_features.h"

#include <cctype>
#include <stdexcept>

namespace human_limit {

namespace {

constexpr const char* kPieceTypeOrder = "pnbrq";

void transform(int& r, int& c, bool isBlack, bool mirror) {
    if (isBlack) r = 7 - r;
    if (mirror) c = 7 - c;
}

chess::Pos findKing(const chess::BoardArray& board, bool wantWhite) {
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;
            if (std::tolower(static_cast<unsigned char>(p)) == 'k' && std::isupper(static_cast<unsigned char>(p)) == wantWhite) {
                return {r, c};
            }
        }
    }
    throw std::runtime_error("king not found");
}

}

PerspectiveContext computePerspectiveContext(const chess::BoardArray& board, bool perspIsWhite) {
    chess::Pos king = findKing(board, perspIsWhite);
    int kr = king.r, kc = king.c;
    transform(kr, kc, !perspIsWhite, false);
    bool mirror = kc >= 4;
    if (mirror) kc = 7 - kc;
    return {perspIsWhite, mirror, kr * 4 + kc};
}

int featureIndexForPiece(const PerspectiveContext& ctx, int r, int c, char piece) {
    if (piece == ' ') return -1;
    char pl = std::tolower(static_cast<unsigned char>(piece));

    bool isWhitePiece = std::isupper(static_cast<unsigned char>(piece));
    bool isOwn = isWhitePiece == ctx.perspIsWhite;

    int pieceTypeIdx;
    if (pl == 'k') {
        if (isOwn) return -1;
        pieceTypeIdx = 10;
    } else {
        int typeIdx = -1;
        for (int i = 0; i < 5; i++) {
            if (kPieceTypeOrder[i] == pl) { typeIdx = i; break; }
        }
        pieceTypeIdx = isOwn ? typeIdx : 5 + typeIdx;
    }

    int tr = r, tc = c;
    transform(tr, tc, !ctx.perspIsWhite, ctx.mirror);
    int sqIdx = tr * 8 + tc;
    return ctx.kingBucket * kSlotsPerBucket + pieceTypeIdx * 64 + sqIdx;
}

namespace {

std::vector<int> perspectiveFeatures(const chess::BoardArray& board, bool perspIsWhite) {
    PerspectiveContext ctx = computePerspectiveContext(board, perspIsWhite);
    std::vector<int> indices;
    indices.reserve(30);
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            int idx = featureIndexForPiece(ctx, r, c, board[r][c]);
            if (idx >= 0) indices.push_back(idx);
        }
    }
    return indices;
}

}

int outputBucketFromPieceCount(int pieceCount) {
    int bucket = (pieceCount - 1) / 4;
    return bucket > kNumOutputBuckets - 1 ? kNumOutputBuckets - 1 : bucket;
}

int computeOutputBucket(const chess::BoardArray& board) {
    int pieceCount = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            if (board[r][c] != ' ') pieceCount++;
        }
    }
    return outputBucketFromPieceCount(pieceCount);
}

namespace {

constexpr const char* kThreatPieceTypeOrder = "pnbrqk";

int threatTypeIndex(char pieceLower) {
    for (int i = 0; i < 6; i++) {
        if (kThreatPieceTypeOrder[i] == pieceLower) return i;
    }
    return -1;
}

bool pieceAttacksSquare(const chess::BoardArray& board, int fr, int fc, int tr, int tc) {
    char piece = board[fr][fc];
    char pl = static_cast<char>(std::tolower(static_cast<unsigned char>(piece)));
    bool isWhite = std::isupper(static_cast<unsigned char>(piece));
    int dr = tr - fr, dc = tc - fc;

    switch (pl) {
        case 'p': {
            int dir = isWhite ? -1 : 1;
            return dr == dir && (dc == 1 || dc == -1);
        }
        case 'n': {
            int adr = dr < 0 ? -dr : dr;
            int adc = dc < 0 ? -dc : dc;
            return (adr == 1 && adc == 2) || (adr == 2 && adc == 1);
        }
        case 'k': {
            int adr = dr < 0 ? -dr : dr;
            int adc = dc < 0 ? -dc : dc;
            return adr <= 1 && adc <= 1 && !(dr == 0 && dc == 0);
        }
        case 'b':
        case 'r':
        case 'q': {
            bool diag = (dr == dc || dr == -dc) && dr != 0;
            bool ortho = (dr == 0) != (dc == 0);
            if (pl == 'b' && !diag) return false;
            if (pl == 'r' && !ortho) return false;
            if (pl == 'q' && !diag && !ortho) return false;
            int stepR = (dr > 0) - (dr < 0);
            int stepC = (dc > 0) - (dc < 0);
            int r = fr + stepR, c = fc + stepC;
            while (r != tr || c != tc) {
                if (board[r][c] != ' ') return false;
                r += stepR;
                c += stepC;
            }
            return true;
        }
        default:
            return false;
    }
}

}

std::vector<ThreatFact> computeThreatFacts(const chess::BoardArray& board) {
    struct Piece {
        int r, c;
        bool isWhite;
        int typeIdx;
    };
    Piece pieces[32];
    int numPieces = 0;
    for (int r = 0; r < 8; r++) {
        for (int c = 0; c < 8; c++) {
            char p = board[r][c];
            if (p == ' ') continue;
            pieces[numPieces++] = {r, c, std::isupper(static_cast<unsigned char>(p)) != 0,
                                    threatTypeIndex(static_cast<char>(std::tolower(static_cast<unsigned char>(p))))};
        }
    }

    bool seen[2][kNumThreatPieceTypes][kNumThreatPieceTypes] = {};
    std::vector<ThreatFact> facts;

    for (int i = 0; i < numPieces; i++) {
        for (int j = 0; j < numPieces; j++) {
            if (pieces[i].isWhite == pieces[j].isWhite) continue;
            if (!pieceAttacksSquare(board, pieces[i].r, pieces[i].c, pieces[j].r, pieces[j].c)) continue;

            int side = pieces[i].isWhite ? 0 : 1;
            if (seen[side][pieces[i].typeIdx][pieces[j].typeIdx]) continue;
            seen[side][pieces[i].typeIdx][pieces[j].typeIdx] = true;

            facts.push_back({pieces[i].isWhite, pieces[i].typeIdx, pieces[j].typeIdx});
        }
    }
    return facts;
}

std::vector<int> threatFeaturesForPerspective(const std::vector<ThreatFact>& facts, int kingBucket,
                                               bool perspIsWhite) {
    std::vector<int> indices;
    indices.reserve(facts.size());
    for (const auto& f : facts) {
        int relation = (f.attackerIsWhite == perspIsWhite) ? 0 : 1;
        int idx = kNumPlacementFeatures + kingBucket * kThreatsPerBucket +
                  relation * kNumThreatPieceTypes * kNumThreatPieceTypes +
                  f.attackerType * kNumThreatPieceTypes + f.victimType;
        indices.push_back(idx);
    }
    return indices;
}

std::vector<int> computeThreatFeatures(const chess::BoardArray& board, bool perspIsWhite) {
    PerspectiveContext ctx = computePerspectiveContext(board, perspIsWhite);
    return threatFeaturesForPerspective(computeThreatFacts(board), ctx.kingBucket, perspIsWhite);
}

FeatureSet extractNnueFeatures(const chess::BoardArray& board, chess::Color sideToMove) {
    bool stmIsWhite = sideToMove == chess::Color::White;
    FeatureSet f;
    f.stm = perspectiveFeatures(board, stmIsWhite);
    f.ntm = perspectiveFeatures(board, !stmIsWhite);
    auto stmThreats = computeThreatFeatures(board, stmIsWhite);
    auto ntmThreats = computeThreatFeatures(board, !stmIsWhite);
    f.stm.insert(f.stm.end(), stmThreats.begin(), stmThreats.end());
    f.ntm.insert(f.ntm.end(), ntmThreats.begin(), ntmThreats.end());
    return f;
}

}
