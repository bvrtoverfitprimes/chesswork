// prints raw_engine eval breakdown for a FEN: total + per-term (white-relative cp)
#include <chrono>
#include <cstdio>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../engine/raw_engine/evaluation.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: raw_eval_cli \"<fen>\"\n");
        return 1;
    }
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();
    chess::bitboard::Position pos(argv[1]);
    if (argc > 2 && std::string(argv[2]) == "bench") {
        volatile int sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        const int N = 300000;
        for (int i = 0; i < N; i++) sink += raw_engine::evaluateWhiteRelative(pos);
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now() - t0).count();
        std::printf("bench: %.1f ns/eval (%d evals, sink %d)\n", double(ns) / N, N, sink);
        return 0;
    }
    raw_engine::EvalBreakdown b;
    raw_engine::evaluateWhiteRelative(pos, &b);
    std::printf("total %d\n", b.total);
    std::printf("material %d\n", b.material);
    std::printf("pst %d\n", b.pst);
    std::printf("mobility %d\n", b.mobility);
    std::printf("kingSafety %d\n", b.kingSafety);
    std::printf("kingAttack %d\n", b.kingAttack);
    std::printf("pawnStructure %d\n", b.pawnStructure);
    std::printf("passedPawns %d\n", b.passedPawns);
    std::printf("pawnShield %d\n", b.pawnShield);
    std::printf("rookFiles %d\n", b.rookFiles);
    std::printf("rookSeventh %d\n", b.rookSeventh);
    std::printf("bishopPair %d\n", b.bishopPair);
    std::printf("bishopMobility %d\n", b.bishopMobility);
    std::printf("knightOutposts %d\n", b.knightOutposts);
    std::printf("threats %d\n", b.threats);
    std::printf("pins %d\n", b.pins);
    std::printf("hangingPieces %d\n", b.hangingPieces);
    std::printf("tacticalPressure %d\n", b.tacticalPressure);
    std::printf("space %d\n", b.space);
    std::printf("centerControl %d\n", b.centerControl);
    std::printf("endgameKing %d\n", b.endgameKing);
    std::printf("pieceQuality %d\n", b.pieceQuality);
    std::printf("tempo %d\n", b.tempo);
    std::printf("phase %d\n", b.phase);
    if (argc > 2) {
        std::printf("--- piece report ---\n");
        raw_engine::printPieceQuality(pos);
    }
    return 0;
}
