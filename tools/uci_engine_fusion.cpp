#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../chess/bitboard/bitboard.h"
#include "../chess/bitboard/magic.h"
#include "../chess/bitboard/position.h"
#include "../engine/fusion_dev/network.h"
#include "../engine/fusion_dev/search.h"

using chess::bitboard::Position;

namespace {

Position buildPosition(const std::vector<std::string>& tokens, size_t start) {
    // tokens[start] is "startpos" or "fen"
    size_t i = start;
    std::unique_ptr<Position> pos;
    if (i < tokens.size() && tokens[i] == "startpos") {
        pos = std::make_unique<Position>();
        i++;
    } else if (i < tokens.size() && tokens[i] == "fen") {
        i++;
        std::string fen;
        for (int f = 0; f < 6 && i < tokens.size() && tokens[i] != "moves"; f++, i++) {
            if (!fen.empty()) fen += " ";
            fen += tokens[i];
        }
        pos = std::make_unique<Position>(fen);
    } else {
        pos = std::make_unique<Position>();
    }

    if (i < tokens.size() && tokens[i] == "moves") {
        i++;
        for (; i < tokens.size(); i++) {
            const std::string& mv = tokens[i];
            if (mv.size() < 4) continue;
            int from = Position::parseSquareUci(mv.substr(0, 2));
            int to = Position::parseSquareUci(mv.substr(2, 2));
            char promo = (mv.size() >= 5) ? mv[4] : 'q';
            pos->makeMove(from, to, promo);
        }
    }
    return *pos;
}

int allocateTime(const std::vector<std::string>& tokens, chess::Color turn) {
    int movetime = 0, wtime = 0, btime = 0, winc = 0, binc = 0, movestogo = 0;
    bool infinite = false;
    for (size_t i = 1; i < tokens.size(); i++) {
        auto next = [&](int& out) { if (i + 1 < tokens.size()) out = std::atoi(tokens[++i].c_str()); };
        if (tokens[i] == "movetime") next(movetime);
        else if (tokens[i] == "wtime") next(wtime);
        else if (tokens[i] == "btime") next(btime);
        else if (tokens[i] == "winc") next(winc);
        else if (tokens[i] == "binc") next(binc);
        else if (tokens[i] == "movestogo") next(movestogo);
        else if (tokens[i] == "infinite") infinite = true;
    }
    if (infinite) return 3'600'000;
    if (movetime > 0) return movetime;
    int remaining = (turn == chess::Color::White) ? wtime : btime;
    int inc = (turn == chess::Color::White) ? winc : binc;
    if (remaining <= 0) return 1000;
    int moves = movestogo > 0 ? movestogo : 30;
    int budget = remaining / moves + inc / 2;
    int cap = remaining * 4 / 5;
    if (budget > cap) budget = cap;
    if (budget < 5) budget = 5;
    return budget;
}

}

int main() {
    std::ios_base::sync_with_stdio(false);
    chess::bitboard::initAttackTables();
    chess::bitboard::initMagics();

    human_limit::Network net;
    const char* rw = std::getenv("RAW_WEIGHT");
    bool rawOnly = rw && std::atof(rw) >= 0.999;
    if (!rawOnly) {
        const char* weightsPath = std::getenv("LIMIT_WEIGHTS");
        net.load(weightsPath ? weightsPath : "engine/limit/nnue_weights.bin");
    }

    int numThreads = 1;
    auto searcher = std::make_unique<human_limit::Searcher>(net);
    Position pos;

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);
        if (tokens.empty()) continue;

        const std::string& cmd = tokens[0];
        if (cmd == "uci") {
            std::cout << "id name human_limit\n";
            std::cout << "id author chesswork\n";
            std::cout << "option name Threads type spin default 1 min 1 max 16\n";
            std::cout << "option name Hash type spin default 128 min 1 max 4096\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "setoption") {
            // "setoption name Threads value N"
            std::string name, valStr;
            for (size_t i = 1; i + 1 < tokens.size(); i++) {
                if (tokens[i] == "name") name = tokens[i + 1];
                else if (tokens[i] == "value") valStr = tokens[i + 1];
            }
            if (name == "Threads" && !valStr.empty()) {
                numThreads = std::max(1, std::atoi(valStr.c_str()));
                searcher->setThreads(numThreads);
            }
        } else if (cmd == "ucinewgame") {
            searcher = std::make_unique<human_limit::Searcher>(net);
            searcher->setThreads(numThreads);
        } else if (cmd == "position") {
            pos = buildPosition(tokens, 1);
        } else if (cmd == "go") {
            int timeMs = allocateTime(tokens, pos.turn());
            auto r = searcher->findBestMove(pos, 64, timeMs);
            std::cout << "bestmove " << r.uci << "\n" << std::flush;
        } else if (cmd == "quit") {
            break;
        }
    }
    return 0;
}
