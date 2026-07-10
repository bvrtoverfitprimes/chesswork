#include "evaluation.h"

#include <cctype>

#include "../../chess/bitboard/bitboard.h"
#include "../../chess/bitboard/magic.h"

namespace raw_engine {

namespace {

using namespace chess::bitboard;

constexpr int kMgVal[6] = {82, 337, 365, 477, 1025, 0};
constexpr int kEgVal[6] = {94, 281, 297, 512, 936, 0};
constexpr int kPhaseW[6] = {0, 1, 1, 2, 4, 0};
constexpr int kMaxPhase = 24;

// PeSTO tables, row 0 = rank 8 (index with sq^56 for White, sq for Black).
const int kMgPST[6][64] = {
    {0,0,0,0,0,0,0,0, 98,134,61,95,68,126,34,-11, -6,7,26,31,65,56,25,-20,
     -14,13,6,21,23,12,17,-23, -27,-2,-5,12,17,6,10,-25, -26,-4,-4,-10,3,3,33,-12,
     -35,-1,-20,-23,-15,24,38,-22, 0,0,0,0,0,0,0,0},
    {-167,-89,-34,-49,61,-97,-15,-107, -73,-41,72,36,23,62,7,-17, -47,60,37,65,84,129,73,44,
     -9,17,19,53,37,69,18,22, -13,4,16,13,28,19,21,-8, -23,-9,12,10,19,17,25,-16,
     -29,-53,-12,-3,-1,18,-14,-19, -105,-21,-58,-33,-17,-28,-19,-23},
    {-29,4,-82,-37,-25,-42,7,-8, -26,16,-18,-13,30,59,18,-47, -16,37,43,40,35,50,37,-2,
     -4,5,19,50,37,37,7,-2, -6,13,13,26,34,12,10,4, 0,15,15,15,14,27,18,10,
     4,15,16,0,7,21,33,1, -33,-3,-14,-21,-13,-12,-39,-21},
    {32,42,32,51,63,9,31,43, 27,32,58,62,80,67,26,44, -5,19,26,36,17,45,61,16,
     -24,-11,7,26,24,35,-8,-20, -36,-26,-12,-1,9,-7,6,-23, -45,-25,-16,-17,3,0,-5,-33,
     -44,-16,-20,-9,-1,11,-6,-71, -19,-13,1,17,16,7,-37,-26},
    {-28,0,29,12,59,44,43,45, -24,-39,-5,1,-16,57,28,54, -13,-17,7,8,29,56,47,57,
     -27,-27,-16,-16,-1,17,-2,1, -9,-26,-9,-10,-2,-4,3,-3, -14,2,-11,-2,-5,2,14,5,
     -35,-8,11,2,8,15,-3,1, -1,-18,-9,10,-15,-25,-31,-50},
    {-65,23,16,-15,-56,-34,2,13, 29,-1,-20,-7,-8,-4,-38,-29, -9,24,2,-16,-20,6,22,-22,
     -17,-20,-12,-27,-30,-25,-14,-36, -49,-1,-27,-39,-46,-44,-33,-51, -14,-14,-22,-46,-44,-30,-15,-27,
     1,7,-8,-64,-43,-16,9,8, -15,36,12,-54,8,-28,24,14},
};

const int kEgPST[6][64] = {
    {0,0,0,0,0,0,0,0, 178,173,158,134,147,132,165,187, 94,100,85,67,56,53,82,84,
     32,24,13,5,-2,4,17,17, 13,9,-3,-7,-7,-8,3,-1, 4,7,-6,1,0,-5,-1,-8,
     13,8,8,10,13,0,2,-7, 0,0,0,0,0,0,0,0},
    {-58,-38,-13,-28,-31,-27,-63,-99, -25,-8,-25,-2,-9,-25,-24,-52, -24,-20,10,9,-1,-9,-19,-41,
     -17,3,22,22,22,11,8,-18, -18,-6,16,25,16,17,4,-18, -23,-3,-1,15,10,-3,-20,-22,
     -42,-20,-10,-5,-2,-20,-23,-44, -29,-51,-23,-15,-22,-18,-50,-64},
    {-14,-21,-11,-8,-7,-9,-17,-24, -8,-4,7,-12,-3,-13,-4,-14, 2,-8,0,-1,-2,6,0,4,
     -3,9,12,9,14,10,3,2, -6,3,13,19,7,10,-3,-9, -12,-3,8,10,13,3,-7,-15,
     -14,-18,-7,-1,4,-9,-15,-27, -23,-9,-23,-5,-9,-16,-5,-17},
    {13,10,18,15,12,12,8,5, 11,13,13,11,-3,3,8,3, 7,7,7,5,4,-3,-5,-3,
     4,3,13,1,2,1,-1,2, 3,5,8,4,-5,-6,-8,-11, -4,0,-5,-1,-7,-12,-8,-16,
     -6,-6,0,2,-9,-9,-11,-3, -9,2,3,-1,-5,-13,4,-20},
    {-9,22,22,27,27,19,10,20, -17,20,32,41,58,25,30,0, -20,6,9,49,47,35,19,9,
     3,22,24,45,57,40,57,36, -18,28,19,47,31,34,39,23, -16,-27,15,6,9,17,10,5,
     -22,-23,-30,-16,-16,-23,-36,-32, -33,-28,-22,-43,-5,-32,-20,-41},
    {-74,-35,-18,-18,-11,15,4,-17, -12,17,14,17,17,38,23,11, 10,17,23,15,20,45,44,13,
     -8,22,24,27,26,33,26,3, -18,-4,21,24,27,23,9,-11, -19,-3,11,21,23,16,7,-9,
     -27,-11,4,13,14,4,-5,-17, -53,-34,-21,-11,-28,-14,-24,-43},
};

constexpr int kMobilityMg[6] = {0, 4, 3, 2, 1, 0};
constexpr int kMobilityEg[6] = {0, 4, 3, 4, 2, 0};
constexpr int kPassedMg[8] = {0, 5, 10, 20, 35, 60, 100, 0};
constexpr int kPassedEg[8] = {0, 10, 20, 35, 60, 100, 160, 0};
// attack units per attacker type touching the enemy king ring; converted to
// cp via the nonlinear kSafetyTable (few attackers ~ nothing, coordinated
// attack ~ decisive), standard classical-engine shape.
constexpr int kAttackUnits[6] = {0, 2, 2, 3, 5, 0};
constexpr int kSafetyTable[64] = {
    0, 0, 1, 2, 3, 5, 7, 9, 12, 15, 18, 22, 26, 30, 35, 39,
    44, 50, 56, 62, 68, 75, 82, 85, 89, 97, 105, 113, 122, 131, 140, 150,
    169, 180, 191, 202, 213, 225, 237, 248, 260, 272, 283, 295, 307, 319, 330, 342,
    354, 366, 377, 389, 401, 412, 424, 436, 448, 459, 471, 483, 494, 500, 500, 500,
};

// squares a knight reaches in <=2 moves from k (occupancy-blind)
Bitboard knight2hop(int k) {
    static Bitboard tbl[64];
    static bool init = false;
    if (!init) {
        for (int s2 = 0; s2 < 64; s2++) {
            Bitboard r = knightAttacks[s2];
            Bitboard one = knightAttacks[s2];
            while (one) r |= knightAttacks[popLsb(one)];
            tbl[s2] = r;
        }
        init = true;
    }
    return tbl[k];
}

// Expected-Time-to-Contact bonuses: [type][tier] tier1 = attacks king ring now,
// tier2 = can attack it next move (graph reachability, not euclidean distance)
constexpr int kContactMg[6][3] = {{0,0,0},{0,24,10},{0,18,8},{0,22,9},{0,30,14},{0,0,0}};
constexpr int kContactEg[6][3] = {{0,0,0},{0,12,5},{0,10,4},{0,12,5},{0,16,7},{0,0,0}};

constexpr Bitboard kCenterCore = 0x0000001818000000ULL;   // d4 e4 d5 e5
constexpr Bitboard kCenterExt  = 0x00003C24243C0000ULL;   // c/f-file ring around core
constexpr int kInfW[6] = {0, 5, 4, 4, 6, 0};

// pawn-structure hash: doubled/isolated score + passer sets depend only on pawns.
// NOTE: single shared table; raw path currently runs single-threaded.
struct PawnEntry { Bitboard wp = ~0ULL, bp = 0; int mgP = 0, egP = 0; Bitboard passers[2] = {0, 0}; };
PawnEntry g_pawnTable[1 << 15];

int pieceTypeIdx(char lower) {
    switch (lower) {
        case 'p': return 0;
        case 'n': return 1;
        case 'b': return 2;
        case 'r': return 3;
        case 'q': return 4;
        default: return 5;
    }
}

Bitboard fileBB(int f) { return kFileA << f; }

Bitboard adjacentFilesBB(int f) {
    Bitboard b = 0;
    if (f > 0) b |= fileBB(f - 1);
    if (f < 7) b |= fileBB(f + 1);
    return b;
}

// squares ahead of sq (from side's perspective) on same+adjacent files
Bitboard passedMask(int sq, bool white) {
    int f = fileOf(sq), r = rankOf(sq);
    Bitboard files = fileBB(f) | adjacentFilesBB(f);
    Bitboard ahead = 0;
    if (white) {
        for (int rr = r + 1; rr < 8; rr++) ahead |= (kRank1 << (8 * rr));
    } else {
        for (int rr = r - 1; rr >= 0; rr--) ahead |= (kRank1 << (8 * rr));
    }
    return files & ahead;
}

}

int evaluateWhiteRelative(const chess::bitboard::Position& pos, EvalBreakdown* bd) {
    Bitboard bb[2][6] = {};
    int mgMat = 0, egMat = 0, mg = 0, eg = 0, phase = 0;

    for (int sq = 0; sq < 64; sq++) {
        char p = pos.pieceAt(sq);
        if (p == ' ') continue;
        bool white = std::isupper(static_cast<unsigned char>(p));
        int t = pieceTypeIdx(static_cast<char>(std::tolower(static_cast<unsigned char>(p))));
        bb[white ? 0 : 1][t] |= (1ULL << sq);
        int ti = white ? (sq ^ 56) : sq;
        if (white) {
            mgMat += kMgVal[t]; egMat += kEgVal[t];
            mg += kMgPST[t][ti]; eg += kEgPST[t][ti];
        } else {
            mgMat -= kMgVal[t]; egMat -= kEgVal[t];
            mg -= kMgPST[t][ti]; eg -= kEgPST[t][ti];
        }
        phase += kPhaseW[t];
    }
    if (phase > kMaxPhase) phase = kMaxPhase;

    Bitboard occ = pos.allOccupiedBitboard();
    Bitboard own[2] = {pos.occupiedBitboard(chess::Color::White),
                        pos.occupiedBitboard(chess::Color::Black)};
    Bitboard pawns[2] = {bb[0][0], bb[1][0]};
    Bitboard allPawns = pawns[0] | pawns[1];
    int kingSq[2] = {pos.kingSquare(chess::Color::White), pos.kingSquare(chess::Color::Black)};
    Bitboard kingRing[2] = {kingAttacks[kingSq[0]] | (1ULL << kingSq[0]),
                             kingAttacks[kingSq[1]] | (1ULL << kingSq[1])};

    Bitboard passers[2] = {0, 0};
    Bitboard vB[2] = {bishopAttacks(kingSq[0], occ), bishopAttacks(kingSq[1], occ)};
    Bitboard vR[2] = {rookAttacks(kingSq[0], occ), rookAttacks(kingSq[1], occ)};
    Bitboard vQ[2] = {vB[0] | vR[0], vB[1] | vR[1]};
    Bitboard pawnAtt[2] = {0, 0};
    {
        Bitboard wp = pawns[0];
        while (wp) pawnAtt[0] |= whitePawnAttacks[popLsb(wp)];
        Bitboard bp = pawns[1];
        while (bp) pawnAtt[1] |= blackPawnAttacks[popLsb(bp)];
    }
    // kings defend their ring too
    Bitboard attackedBy[2] = {pawnAtt[0] | kingAttacks[kingSq[0]],
                               pawnAtt[1] | kingAttacks[kingSq[1]]};

    int mgMob = 0, egMob = 0, mgBMob = 0, egBMob = 0, mgPQ = 0, egPQ = 0;
    int mgOut = 0, egOut = 0, mgCtr = 0, egCtr = 0;
    int mgPawn = 0, egPawn = 0, mgPassed = 0, egPassed = 0, mgKing = 0;
    int mgRook = 0, egRook = 0, mgBP = 0, egBP = 0, mgShield = 0;
    int mgThreat = 0, egThreat = 0, mgHang = 0, egHang = 0;

    {
        uint64_t pk = pawns[0] * 0x9E3779B97F4A7C15ULL ^ pawns[1] * 0xC2B2AE3D27D4EB4FULL;
        PawnEntry& pe = g_pawnTable[pk >> 49];
        if (pe.wp != pawns[0] || pe.bp != pawns[1]) {
            pe.wp = pawns[0]; pe.bp = pawns[1];
            pe.mgP = pe.egP = 0; pe.passers[0] = pe.passers[1] = 0;
            for (int ps = 0; ps < 2; ps++) {
                int psgn = ps == 0 ? 1 : -1;
                Bitboard pp = pawns[ps];
                while (pp) {
                    int sq = popLsb(pp);
                    int f = fileOf(sq);
                    if (popcount(pawns[ps] & fileBB(f)) > 1) { pe.mgP -= psgn * 8; pe.egP -= psgn * 14; }
                    if (!(pawns[ps] & adjacentFilesBB(f))) { pe.mgP -= psgn * 12; pe.egP -= psgn * 9; }
                    if (!(pawns[ps == 0 ? 1 : 0] & passedMask(sq, ps == 0))) pe.passers[ps] |= (1ULL << sq);
                }
            }
        }
        mgPawn += pe.mgP;
        egPawn += pe.egP;
        passers[0] = pe.passers[0];
        passers[1] = pe.passers[1];
    }

    for (int side = 0; side < 2; side++) {
        int sgn = side == 0 ? 1 : -1;
        int enemy = side ^ 1;
        int attackUnits = 0;
        int attackerCount = 0;
        Bitboard mobArea = ~own[side] & ~pawnAtt[enemy];

        if (popcount(bb[side][2]) >= 2) { mgBP += sgn * 25; egBP += sgn * 45; }

        // pawns
        Bitboard p = pawns[side];
        while (p) {
            int sq = popLsb(p);
            int f = fileOf(sq);
            if (popcount(pawns[side] & fileBB(f)) > 1) { mgPawn -= sgn * 8; egPawn -= sgn * 14; }
            if (!(pawns[side] & adjacentFilesBB(f))) { mgPawn -= sgn * 12; egPawn -= sgn * 9; }

        }

        // knights
        Bitboard n = bb[side][1];
        while (n) {
            int sq = popLsb(n);
            Bitboard att = knightAttacks[sq];
            attackedBy[side] |= att;
            int mob = popcount(att & mobArea);
            mgMob += sgn * kMobilityMg[1] * (mob - 4);
            egMob += sgn * kMobilityEg[1] * (mob - 4);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[1]; attackerCount++; }
            {
                int inf = 2 * popcount(att & kingRing[enemy]) +
                          popcount(knight2hop(sq) & kingRing[enemy]) +
                          2 * popcount(att & kingRing[enemy] & pawnAtt[side]);
                int v = kInfW[1] * inf; if (v > 45) v = 45;
                mgPQ += sgn * v; egPQ += sgn * v / 2;
            }
            mgCtr += sgn * (3 * popcount(att & kCenterCore) + popcount(att & kCenterExt));
            egCtr += sgn * popcount(att & kCenterCore);
            if (pawnAtt[side] & (1ULL << sq)) { mgPQ += sgn * 8; egPQ += sgn * 4; }
            {
                int defended = popcount(att & own[side] & ~pawns[side]);
                mgPQ += sgn * 3 * (defended > 3 ? 3 : defended);
            }
            int rel = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
            if (rel >= 3 && rel <= 5 && (pawnAtt[side] & (1ULL << sq)) &&
                !(pawns[enemy] & passedMask(sq, side == 0) & adjacentFilesBB(fileOf(sq)))) {
                mgOut += sgn * 28; egOut += sgn * 16;
            }
            {
                int relr = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
                if (mob <= 1 && relr > 0) { mgPQ -= sgn * 40; egPQ -= sgn * 30; }
            }
        }
        // bishops
        Bitboard b = bb[side][2];
        while (b) {
            int sq = popLsb(b);
            Bitboard att = bishopAttacks(sq, occ);
            attackedBy[side] |= att;
            int mob = popcount(att & mobArea);
            mgBMob += sgn * kMobilityMg[2] * (mob - 6);
            egBMob += sgn * kMobilityEg[2] * (mob - 6);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[2]; attackerCount++; }
            {
                int inf = 2 * popcount(att & kingRing[enemy]) + popcount(att & vB[enemy]) +
                          2 * popcount(att & kingRing[enemy] & pawnAtt[side]);
                int v = kInfW[2] * inf; if (v > 45) v = 45;
                mgPQ += sgn * v; egPQ += sgn * v / 2;
            }
            mgCtr += sgn * (3 * popcount(att & kCenterCore) + popcount(att & kCenterExt));
            egCtr += sgn * popcount(att & kCenterCore);
            {
                int defended = popcount(att & own[side] & ~pawns[side]);
                mgPQ += sgn * 3 * (defended > 3 ? 3 : defended);
            }
            {
                int relr = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
                if (mob <= 1 && relr > 0) { mgPQ -= sgn * 40; egPQ -= sgn * 30; }
            }
        }
        // rooks
        Bitboard r = bb[side][3];
        while (r) {
            int sq = popLsb(r);
            Bitboard att = rookAttacks(sq, occ);
            attackedBy[side] |= att;
            int mob = popcount(att & mobArea);
            mgMob += sgn * kMobilityMg[3] * (mob - 7);
            egMob += sgn * kMobilityEg[3] * (mob - 7);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[3]; attackerCount++; }
            {
                int inf = 2 * popcount(att & kingRing[enemy]) + popcount(att & vR[enemy]) +
                          2 * popcount(att & kingRing[enemy] & pawnAtt[side]);
                int v = kInfW[3] * inf; if (v > 45) v = 45;
                mgPQ += sgn * v; egPQ += sgn * v / 2;
            }
            if (att & (bb[side][3] | bb[side][4])) { mgPQ += sgn * 10; egPQ += sgn * 5; }  // battery/connected
            {
                int relr = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
                if (mob <= 1 && relr > 0) { mgPQ -= sgn * 30; egPQ -= sgn * 20; }
            }
            int f = fileOf(sq);
            if (!(allPawns & fileBB(f))) { mgRook += sgn * 25; egRook += sgn * 12; }
            else if (!(pawns[side] & fileBB(f))) { mgRook += sgn * 12; egRook += sgn * 6; }
        }
        // queens
        Bitboard q = bb[side][4];
        while (q) {
            int sq = popLsb(q);
            Bitboard att = queenAttacks(sq, occ);
            attackedBy[side] |= att;
            int mob = popcount(att & mobArea);
            mgMob += sgn * kMobilityMg[4] * (mob - 13);
            egMob += sgn * kMobilityEg[4] * (mob - 13);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[4]; attackerCount++; }
            {
                int inf = 2 * popcount(att & kingRing[enemy]) + popcount(att & vQ[enemy]) +
                          2 * popcount(att & kingRing[enemy] & pawnAtt[side]);
                int v = kInfW[4] * inf; if (v > 60) v = 60;
                mgPQ += sgn * v; egPQ += sgn * v / 2;
            }
        }
        // king safety: nonlinear in total attack units; single attacker ~ harmless
        if (attackerCount >= 2 && bb[side][4]) {
            int idx = attackUnits * attackerCount / 2;
            if (idx > 63) idx = 63;
            mgKing += sgn * kSafetyTable[idx] * phase / kMaxPhase;
        }

        // pawn shield in front of own king (mg only)
        int kf = fileOf(kingSq[side]);
        Bitboard shieldZone = (fileBB(kf) | adjacentFilesBB(kf));
        int kr = rankOf(kingSq[side]);
        Bitboard nearRanks = 0;
        for (int rr = kr - 2; rr <= kr + 2; rr++)
            if (rr >= 0 && rr < 8) nearRanks |= (kRank1 << (8 * rr));
        int shield = popcount(pawns[side] & shieldZone & nearRanks &
                              (side == 0 ? passedMask(kingSq[side], true)
                                         : passedMask(kingSq[side], false)));
        mgShield += sgn * 10 * (shield > 3 ? 3 : shield);
    }

    // passed pawns: bonus reduced for blockaded/contested stop squares
    for (int side = 0; side < 2; side++) {
        int sgn = side == 0 ? 1 : -1;
        int enemy = side ^ 1;
        Bitboard p = passers[side];
        while (p) {
            int sq = popLsb(p);
            int rel = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
            int mgB = kPassedMg[rel], egB = kPassedEg[rel];
            int stop = side == 0 ? sq + 8 : sq - 8;
            if (stop >= 0 && stop < 64) {
                if (own[enemy] & (1ULL << stop)) { mgB /= 2; egB /= 2; }          // blockaded
                else if ((attackedBy[enemy] & (1ULL << stop)) &&
                         !(attackedBy[side] & (1ULL << stop))) { mgB = mgB * 2 / 3; egB = egB * 2 / 3; }
            }
            mgPassed += sgn * mgB;
            egPassed += sgn * egB;
        }
    }

    // threats: minor/major attacked by an enemy pawn; any non-pawn attacked and undefended
    for (int side = 0; side < 2; side++) {
        int sgn = side == 0 ? 1 : -1;
        int enemy = side ^ 1;
        Bitboard nonPawn = own[side] & ~pawns[side] & ~(1ULL << kingSq[side]);
        int pawnThreatened = popcount(nonPawn & pawnAtt[enemy]);
        mgThreat -= sgn * 40 * pawnThreatened;
        egThreat -= sgn * 30 * pawnThreatened;
        Bitboard hangingBB = nonPawn & attackedBy[enemy] & ~attackedBy[side];
        while (hangingBB) {
            int hsq = popLsb(hangingBB);
            int ht = pieceTypeIdx(static_cast<char>(std::tolower(
                static_cast<unsigned char>(pos.pieceAt(hsq)))));
            // scaled by victim value: losing a hanging queen is not 22cp
            mgHang -= sgn * kMgVal[ht] / 8;
            egHang -= sgn * kEgVal[ht] / 8;
        }
    }

    auto taper = [&](int m, int e) { return (m * phase + e * (kMaxPhase - phase)) / kMaxPhase; };
    int tempoVal = (pos.turn() == chess::Color::White) ? 12 : -12;
    int material = taper(mgMat, egMat);
    int pst = taper(mg, eg);
    int mob = taper(mgMob, egMob);
    int bmob = taper(mgBMob, egBMob);
    int pawnS = taper(mgPawn, egPawn);
    int passed = taper(mgPassed, egPassed);
    int thr = taper(mgThreat, egThreat);
    int hang = taper(mgHang, egHang);
    int rookF = taper(mgRook, egRook);
    int bp = taper(mgBP, egBP);
    int kingA = taper(mgKing, 0);
    int shieldS = taper(mgShield, 0);
    int pq = taper(mgPQ, egPQ);
    int outp = taper(mgOut, egOut);
    int ctr = taper(mgCtr, egCtr);
    int score = material + pst + mob + bmob + pawnS + passed + thr + hang +
                rookF + bp + kingA + shieldS + pq + outp + ctr + tempoVal;
    if (bd) {
        bd->material = material; bd->pst = pst;
        bd->mobility = mob; bd->bishopMobility = bmob;
        bd->kingAttack = kingA; bd->pawnShield = shieldS;
        bd->pawnStructure = pawnS; bd->passedPawns = passed;
        bd->threats = thr; bd->hangingPieces = hang;
        bd->rookFiles = rookF; bd->bishopPair = bp;
        bd->pieceQuality = pq; bd->knightOutposts = outp; bd->centerControl = ctr;
        bd->tempo = tempoVal; bd->phase = phase; bd->total = score;
    }
    return score;
}


void printPieceQuality(const chess::bitboard::Position& pos) {
    using namespace chess::bitboard;
    Bitboard occ = pos.allOccupiedBitboard();
    int kingSq[2] = {pos.kingSquare(chess::Color::White), pos.kingSquare(chess::Color::Black)};
    Bitboard kingRing[2] = {kingAttacks[kingSq[0]] | (1ULL << kingSq[0]),
                             kingAttacks[kingSq[1]] | (1ULL << kingSq[1])};
    Bitboard pawns[2] = {0, 0}, ownAll[2] = {pos.occupiedBitboard(chess::Color::White),
                                              pos.occupiedBitboard(chess::Color::Black)};
    for (int sq = 0; sq < 64; sq++) {
        char pc = pos.pieceAt(sq);
        if (pc == 'P') pawns[0] |= 1ULL << sq;
        if (pc == 'p') pawns[1] |= 1ULL << sq;
    }
    Bitboard pawnAtt[2] = {0, 0};
    Bitboard wp = pawns[0]; while (wp) pawnAtt[0] |= whitePawnAttacks[popLsb(wp)];
    Bitboard bp = pawns[1]; while (bp) pawnAtt[1] |= blackPawnAttacks[popLsb(bp)];

    std::printf("piece sq material mobility contact outpost trapped battery utility\n");
    for (int sq = 0; sq < 64; sq++) {
        char pc = pos.pieceAt(sq);
        if (pc == ' ' || pc == 'K' || pc == 'k' || pc == 'P' || pc == 'p') continue;
        bool white = std::isupper(static_cast<unsigned char>(pc));
        int side = white ? 0 : 1, enemy = side ^ 1;
        int t = pieceTypeIdx(static_cast<char>(std::tolower(static_cast<unsigned char>(pc))));
        Bitboard att = 0;
        if (t == 1) att = knightAttacks[sq];
        else if (t == 2) att = bishopAttacks(sq, occ);
        else if (t == 3) att = rookAttacks(sq, occ);
        else att = queenAttacks(sq, occ);
        Bitboard mobArea = ~ownAll[side] & ~pawnAtt[enemy];
        int mob = popcount(att & mobArea);
        int mobBonusBase[6] = {0, 4, 3, 2, 1, 0};
        int mobRef[6] = {0, 4, 6, 7, 13, 0};
        int mobBonus = mobBonusBase[t] * (mob - mobRef[t]);
        int contact = 0;
        if (att & kingRing[enemy]) contact = kContactMg[t][1];
        else {
            Bitboard reach2 = 0;
            if (t == 1) reach2 = knight2hop(sq);
            else if (t == 2) reach2 = att & bishopAttacks(kingSq[enemy], occ);
            else if (t == 3) reach2 = att & rookAttacks(kingSq[enemy], occ);
            else reach2 = att & queenAttacks(kingSq[enemy], occ);
            if (t == 1 ? (reach2 & kingRing[enemy]) != 0 : reach2 != 0) contact = kContactMg[t][2];
        }
        int outpost = 0;
        if (t == 1) {
            int rel = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
            if (rel >= 3 && rel <= 5 && (pawnAtt[side] & (1ULL << sq)) &&
                !(pawns[enemy] & passedMask(sq, side == 0) & adjacentFilesBB(fileOf(sq))))
                outpost = 28;
        }
        int relr = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
        int trapped = (t <= 3 && mob <= 1 && relr > 0) ? (t == 3 ? -30 : -40) : 0;
        int battery = 0;
        // batteries only tracked for rooks in the hot path
        int base = kMgVal[t];
        int utility = base + mobBonus + contact + outpost + trapped + battery;
        std::printf("%c %s %d %+d %+d %+d %+d %+d = %d\n",
                    pc, Position::squareToUci(sq).c_str(), base, mobBonus, contact,
                    outpost, trapped, battery, utility);
    }
}

}
