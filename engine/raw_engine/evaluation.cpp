#include "evaluation.h"

#include <cctype>
#include <cstdlib>

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


// Berserk 4.7 tuned king-safety + threats weights (their scale ~= ours, pawn 100)
constexpr int kKsAttW[6] = {0, 33, 32, 19, 25, 0};   // by type: p,N,B,R,Q,k
constexpr int kKsKnightChk = 279, kKsBishopChk = 311, kKsRookChk = 272, kKsQueenChk = 213;
constexpr int kKsUnsafeChk = 57, kKsWeakSq = 78, kKsNoQueen = -190, kKsKnightDef = -87;
// threat[victim type p,N,B,R,Q,k] = {mg,eg}
constexpr int kKnThrMg[6] = {0, -5, 38, 94, 80, 0};
constexpr int kKnThrEg[6] = {22, 54, 44, 16, -53, 0};
constexpr int kBiThrMg[6] = {4, 26, -66, 78, 69, 0};
constexpr int kBiThrEg[6] = {23, 42, 81, 25, 25, 0};
constexpr int kRkThrMg[6] = {0, 34, 38, 5, 56, 0};
constexpr int kRkThrEg[6] = {26, 49, 63, 21, -50, 0};
constexpr Bitboard kLightSq = 0x55AA55AA55AA55AAULL;

// pawn-structure hash: doubled/isolated score + passer sets depend only on pawns.
// NOTE: single shared table; raw path currently runs single-threaded.
struct PawnEntry { Bitboard wp = ~0ULL, bp = 0; int mgP = 0, egP = 0; Bitboard passers[2] = {0, 0}; };
PawnEntry g_pawnTable[1 << 15];


// runtime category scales (/128). default identity -> byte-identical to untuned.
struct TuneScales { int s[12]; };
const TuneScales& tuneScales() {
    static TuneScales t = [] {
        TuneScales r; for (int i = 0; i < 12; i++) r.s[i] = 128;
        if (const char* e = std::getenv("RAW_TUNE")) {
            const char* p2 = e; int i = 0;
            while (*p2 && i < 12) {
                while (*p2 == ' ') p2++;
                if (!*p2) break;
                int v = std::atoi(p2); r.s[i++] = v;
                while (*p2 && *p2 != ' ') p2++;
            }
        }
        return r;
    }();
    return t;
}

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
    Bitboard pawnAtt[2] = {0, 0};
    {
        Bitboard wp = pawns[0];
        while (wp) pawnAtt[0] |= whitePawnAttacks[popLsb(wp)];
        Bitboard bp = pawns[1];
        while (bp) pawnAtt[1] |= blackPawnAttacks[popLsb(bp)];
    }
    // pawn double-attacks (for two-attack maps)
    Bitboard pawnAtt2[2];
    {
        Bitboard wl = (pawns[0] & ~kFileA) << 7, wr = (pawns[0] & ~kFileH) << 9;
        Bitboard bl = (pawns[1] & ~kFileA) >> 9, br = (pawns[1] & ~kFileH) >> 7;
        pawnAtt2[0] = wl & wr;
        pawnAtt2[1] = bl & br;
    }
    // per-type attack unions [side][0=p,1=N,2=B,3=R,4=Q,5=k], all-attacks, two-attacks
    Bitboard pieceAtt[2][6] = {};
    Bitboard twoAtt[2], attackedBy[2];
    int ksWeight[2] = {0, 0}, ksCount[2] = {0, 0};
    for (int S = 0; S < 2; S++) {
        Bitboard kAtt = kingAttacks[kingSq[S]];
        pieceAtt[S][0] = pawnAtt[S];
        pieceAtt[S][5] = kAtt;
        attackedBy[S] = pawnAtt[S] | kAtt;
        twoAtt[S] = pawnAtt2[S] | (pawnAtt[S] & kAtt);
    }

    int mgMob = 0, egMob = 0, mgBMob = 0, egBMob = 0, mgPQ = 0, egPQ = 0;
    int egEndK = 0;
    int mgPawn = 0, egPawn = 0, mgPassed = 0, egPassed = 0, mgKing = 0;
    int mgRook = 0, egRook = 0, mgBP = 0, egBP = 0, mgShield = 0, egKingS = 0;
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
                    bool hasNeighbor = pawns[ps] & adjacentFilesBB(f);
                    if (!hasNeighbor) { pe.mgP -= psgn * 12; pe.egP -= psgn * 9; }
                    // supported: friendly pawns defending this pawn (Berserk DEFENDED_PAWN 13/10 halved)
                    Bitboard pdef = pawns[ps] & (ps == 0 ? blackPawnAttacks[sq] : whitePawnAttacks[sq]);
                    int nd = popcount(pdef);
                    pe.mgP += psgn * 6 * nd; pe.egP += psgn * 5 * nd;
                    // phalanx: adjacent-file same-rank friendly pawn (rank-scaled, halved shape)
                    if (pawns[ps] & adjacentFilesBB(f) & (kRank1 << (8 * rankOf(sq)))) {
                        int rr = ps == 0 ? rankOf(sq) : 7 - rankOf(sq);
                        static const int cMg[8] = {0, 0, 8, 7, 3, 3, 1, 0};
                        static const int cEg[8] = {0, 0, 12, 7, 2, 2, 0, 0};
                        pe.mgP += psgn * cMg[rr]; pe.egP += psgn * cEg[rr];
                    }
                    // backward: advance square attacked by enemy pawn, not defended by a friendly pawn
                    if (hasNeighbor) {
                        int adv = ps == 0 ? sq + 8 : sq - 8;
                        if (adv >= 0 && adv < 64) {
                            Bitboard advAtk = (ps == 0 ? blackPawnAttacks[adv] : whitePawnAttacks[adv]) & pawns[ps == 0 ? 1 : 0];
                            Bitboard advDef = pawns[ps] & adjacentFilesBB(f) & passedMask(sq, ps == 0);
                            if (advAtk && !advDef) { pe.mgP -= psgn * 4; pe.egP -= psgn * 9; }
                        }
                    }
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
        // minor behind a pawn (Berserk MINOR_BEHIND_PAWN 6/14 halved): shelter/support
        {
            Bitboard minors = bb[side][1] | bb[side][2];
            Bitboard behind = minors & (side == 0 ? (allPawns >> 8) : (allPawns << 8));
            int nb = popcount(behind);
            mgBP += sgn * 3 * nb; egBP += sgn * 7 * nb;
        }

        // knights
        Bitboard n = bb[side][1];
        while (n) {
            int sq = popLsb(n);
            Bitboard att = knightAttacks[sq];
            twoAtt[side] |= attackedBy[side] & att; attackedBy[side] |= att; pieceAtt[side][1] |= att;
            if (att & kingRing[enemy]) { ksWeight[side] += kKsAttW[1]; ksCount[side]++; }
            int mob = popcount(att & mobArea);
            mgMob += sgn * kMobilityMg[1] * (mob - 4);
            egMob += sgn * kMobilityEg[1] * (mob - 4);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[1]; attackerCount++; }
            // knight one move away from attacking the enemy king ring: rim PSTs
            // can't see concrete approach maneuvers (Na5-b3 class)
            else if (knight2hop(sq) & kingRing[enemy]) { mgPQ += sgn * 12; egPQ += sgn * 6; }
        }
        // bishops
        Bitboard b = bb[side][2];
        while (b) {
            int sq = popLsb(b);
            Bitboard att = bishopAttacks(sq, occ);
            twoAtt[side] |= attackedBy[side] & att; attackedBy[side] |= att; pieceAtt[side][2] |= att;
            if (att & kingRing[enemy]) { ksWeight[side] += kKsAttW[2]; ksCount[side]++; }
            int mob = popcount(att & mobArea);
            mgBMob += sgn * kMobilityMg[2] * (mob - 6);
            egBMob += sgn * kMobilityEg[2] * (mob - 6);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[2]; attackerCount++; }
        }
        // rooks
        Bitboard r = bb[side][3];
        while (r) {
            int sq = popLsb(r);
            Bitboard att = rookAttacks(sq, occ);
            twoAtt[side] |= attackedBy[side] & att; attackedBy[side] |= att; pieceAtt[side][3] |= att;
            if (att & kingRing[enemy]) { ksWeight[side] += kKsAttW[3]; ksCount[side]++; }
            int mob = popcount(att & mobArea);
            mgMob += sgn * kMobilityMg[3] * (mob - 7);
            egMob += sgn * kMobilityEg[3] * (mob - 7);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[3]; attackerCount++; }
            int f = fileOf(sq);
            if (!(allPawns & fileBB(f))) { mgRook += sgn * 25; egRook += sgn * 12; }
            else if (!(pawns[side] & fileBB(f))) { mgRook += sgn * 12; egRook += sgn * 6; }
            // rook trapped behind an uncastled king (classic Rh1/Kg1, Ra1/Kb1)
            int kf = fileOf(kingSq[side]);
            int relr = side == 0 ? rankOf(sq) : 7 - rankOf(sq);
            int krelr = side == 0 ? rankOf(kingSq[side]) : 7 - rankOf(kingSq[side]);
            if (relr == 0 && krelr == 0) {
                if ((f > kf && kf >= 4 && kf <= 6) || (f < kf && kf >= 1 && kf <= 3)) {
                    mgRook -= sgn * 21; egRook -= sgn * 15;
                }
            }
        }
        // queens
        Bitboard q = bb[side][4];
        while (q) {
            int sq = popLsb(q);
            Bitboard att = queenAttacks(sq, occ);
            twoAtt[side] |= attackedBy[side] & att; attackedBy[side] |= att; pieceAtt[side][4] |= att;
            if (att & kingRing[enemy]) { ksWeight[side] += kKsAttW[4]; ksCount[side]++; }
            int mob = popcount(att & mobArea);
            mgMob += sgn * kMobilityMg[4] * (mob - 13);
            egMob += sgn * kMobilityEg[4] * (mob - 13);
            if (att & kingRing[enemy]) { attackUnits += kAttackUnits[4]; attackerCount++; }
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

                // endgame technique (eg-only): kings escort passers; control of
                // the promotion path counts. The permanently-zero endgameKing row
                // was indicted three times before this went in.
                auto cheb = [](int a, int b) {
                    int df = fileOf(a) - fileOf(b); if (df < 0) df = -df;
                    int dr = rankOf(a) - rankOf(b); if (dr < 0) dr = -dr;
                    return df > dr ? df : dr;
                };
                egEndK += sgn * 6 * (cheb(kingSq[enemy], stop) - cheb(kingSq[side], stop));
                Bitboard path = 0;
                for (int t2 = stop; t2 >= 0 && t2 < 64; t2 += (side == 0 ? 8 : -8)) path |= (1ULL << t2);
                egEndK += sgn * 4 * (popcount(path & attackedBy[side]) -
                                      popcount(path & attackedBy[enemy]));
            }
            mgPassed += sgn * mgB;
            egPassed += sgn * egB;
        }
    }

    // King safety with SAFE CHECKS (Berserk 4.7 formula + tuned weights). Defender S,
    // attacker X. danger accumulates attacker weight*count, safe checks by type,
    // weak king-ring squares, no-enemy-queen and knight-defender adjustments; then
    // a nonlinear danger^2/1024 penalty (the single biggest classical KS lever).
    for (int S = 0; S < 2; S++) {
        int X = S ^ 1;
        int ksq = kingSq[S];
        Bitboard kArea = kingRing[S];
        Bitboard weak = attackedBy[X] & ~twoAtt[S] &
                        (~attackedBy[S] | pieceAtt[S][4] | pieceAtt[S][5]);
        Bitboard vuln = (~attackedBy[S] | (weak & twoAtt[X])) & ~own[X];
        Bitboard bA = bishopAttacks(ksq, occ), rA = rookAttacks(ksq, occ);
        Bitboard nChk = knightAttacks[ksq] & pieceAtt[X][1] & ~own[X];
        Bitboard bChk = bA & pieceAtt[X][2] & ~own[X];
        Bitboard rChk = rA & pieceAtt[X][3] & ~own[X];
        Bitboard qChk = (bA | rA) & pieceAtt[X][4] & ~own[X];
        int unsafe = popcount((nChk | bChk | rChk) & ~vuln);
        int danger = ksWeight[X] * ksCount[X]
                     + kKsKnightChk * popcount(nChk & vuln)
                     + kKsBishopChk * popcount(bChk & vuln)
                     + kKsRookChk   * popcount(rChk & vuln)
                     + kKsQueenChk  * popcount(qChk & vuln)
                     + kKsUnsafeChk * unsafe
                     + kKsWeakSq    * popcount(weak & kArea)
                     + kKsNoQueen   * (bb[X][4] == 0)
                     + kKsKnightDef * ((pieceAtt[S][1] & kArea) != 0);
        if (danger > 0) {
            int sgnDef = (S == 0) ? -1 : 1;   // danger to S hurts S (white-relative)
            mgKing += sgnDef * danger * danger / 2048;
            egKingS += sgnDef * danger / 64;
        }
    }

    // Threats (Berserk 4.7): threat-by-knight/bishop/rook per victim type, plus the
    // big safe-pawn threat and hanging pieces. weak = squares we attack that the
    // enemy does not adequately cover.
    for (int side = 0; side < 2; side++) {
        int sgn = side == 0 ? 1 : -1;
        int X = side ^ 1;
        Bitboard covered = pieceAtt[X][0] | (twoAtt[X] & ~twoAtt[side]);
        Bitboard nonPawnEnemies = own[X] & ~pawns[X];
        Bitboard weak = ~covered & attackedBy[side];
        auto victimIdx = [&](int sq) {
            return pieceTypeIdx(static_cast<char>(std::tolower(
                static_cast<unsigned char>(pos.pieceAt(sq)))));
        };
        Bitboard t;
        t = pieceAtt[side][1] & own[X] & (nonPawnEnemies | weak);
        while (t) { int v = victimIdx(popLsb(t)); mgThreat += sgn * kKnThrMg[v] / 2; egThreat += sgn * kKnThrEg[v] / 2; }
        t = pieceAtt[side][2] & own[X] & (nonPawnEnemies | weak);
        while (t) { int v = victimIdx(popLsb(t)); mgThreat += sgn * kBiThrMg[v] / 2; egThreat += sgn * kBiThrEg[v] / 2; }
        t = pieceAtt[side][3] & own[X] & weak;
        while (t) { int v = victimIdx(popLsb(t)); mgThreat += sgn * kRkThrMg[v] / 2; egThreat += sgn * kRkThrEg[v] / 2; }
        // safe pawn threat (big): a pawn attacks an enemy non-pawn
        int pawnThr = popcount(nonPawnEnemies & pawnAtt[side]);
        mgThreat += sgn * 47 * pawnThr; egThreat += sgn * 22 * pawnThr;
        // hanging: enemy pieces we attack that they don't defend at all
        Bitboard hangingBB = own[X] & ~attackedBy[X] & attackedBy[side];
        int hangCnt = popcount(hangingBB);
        mgHang += sgn * 5 * hangCnt; egHang += sgn * 12 * hangCnt;
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
    int kingA = taper(mgKing, egKingS);
    int shieldS = taper(mgShield, 0);
    int pq = taper(mgPQ, egPQ);
    int endK = taper(0, egEndK);
    const TuneScales& ts = tuneScales();
    mob = mob * ts.s[0] / 128; bmob = bmob * ts.s[1] / 128;
    kingA = kingA * ts.s[2] / 128;
    int thrhang = (thr + hang) * ts.s[3] / 128;
    passed = passed * ts.s[4] / 128; pawnS = pawnS * ts.s[5] / 128;
    pq = pq * ts.s[6] / 128; rookF = rookF * ts.s[7] / 128;
    bp = bp * ts.s[8] / 128; endK = endK * ts.s[9] / 128;
    int score = material + pst + mob + bmob + pawnS + passed + thrhang +
                rookF + bp + kingA + shieldS + pq + endK + tempoVal;
    // opposite-colored-bishop endgame scaling: OCB endings are very drawish
    if (popcount(bb[0][2]) == 1 && popcount(bb[1][2]) == 1 &&
        ((bb[0][2] & kLightSq) != 0) != ((bb[1][2] & kLightSq) != 0)) {
        bool minorsOnly = (bb[0][1] | bb[1][1] | bb[0][3] | bb[1][3] | bb[0][4] | bb[1][4]) == 0;
        int scale = minorsOnly ? 64 : 96;
        score = score * scale / 128;
    }
    if (bd) {
        bd->material = material; bd->pst = pst;
        bd->mobility = mob; bd->bishopMobility = bmob;
        bd->kingAttack = kingA; bd->pawnShield = shieldS;
        bd->pawnStructure = pawnS; bd->passedPawns = passed;
        bd->threats = thr; bd->hangingPieces = hang;
        bd->rookFiles = rookF; bd->bishopPair = bp;
        bd->pieceQuality = pq;
        bd->endgameKing = endK;
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
