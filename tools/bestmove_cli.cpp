#include <iostream>
#include <string>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../chess/board.h"
#include "../engine/ancient_engine/search.h"
#include "../engine/old_engine/search.h"
#include "../engine/human_limit/network.h"
#include "../engine/human_limit/search.h"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: bestmove_cli <ancient|old|human> <fen> <time_ms>\n";
        return 1;
    }

    std::string which = argv[1];
    int timeMs = std::atoi(argv[3]);

    if (which == "ancient") {
        chess::Game game(argv[2]);
        int depth = 3;
        auto r = ancient_engine::findBestMove(game, depth);
        std::cout << r.uci << std::endl;
    } else if (which == "old") {
        chess::Game game(argv[2]);
        old_engine::Searcher searcher;
        auto r = searcher.findBestMove(game, 64, timeMs);
        std::cout << r.uci << std::endl;
    } else if (which == "human") {
        static bool magicsInit = [] {
            chess::bitboard::initAttackTables();
            chess::bitboard::initMagics();
            return true;
        }();
        (void)magicsInit;
        static human_limit::Network net;
        static bool loaded = net.load("engine/human_limit/nnue_weights.bin");
        (void)loaded;
        chess::bitboard::Position pos(argv[2]);
        human_limit::Searcher searcher(net);
        auto r = searcher.findBestMove(pos, 64, timeMs);
        std::cout << r.uci << std::endl;
    } else {
        std::cerr << "unknown engine: " << which << "\n";
        return 1;
    }

    return 0;
}
