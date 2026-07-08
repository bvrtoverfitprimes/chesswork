#include <iostream>
#include <string>

#include "../chess/board.h"
#include "../engine/human_limit/network.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: eval_cli <fen> [weights_path]\n";
        return 1;
    }

    chess::Game game(argv[1]);
    std::string weightsPath = argc > 2 ? argv[2] : "engine/human_limit/nnue_weights.bin";

    static human_limit::Network net;
    static bool loaded = net.load(weightsPath);
    if (!loaded) {
        std::cerr << "failed to load weights from " << weightsPath << "\n";
        return 1;
    }

    std::cout << net.evaluate(game) << std::endl;
    return 0;
}
