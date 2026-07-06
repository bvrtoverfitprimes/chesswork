#pragma once

#include <string>

#include "../../chess/board.h"

namespace old_engine {

struct SearchResult {
    std::string uci;
    int score;
};

SearchResult findBestMove(chess::Game& game, int depth);

}
