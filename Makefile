CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -flto -ffast-math

CHESS_SRC = chess/board.cpp chess/pieces.cpp
BITBOARD_SRC = chess/bitboard/bitboard.cpp chess/bitboard/magic.cpp chess/bitboard/position.cpp
HUMAN_LIMIT_SRC = engine/human_limit/network.cpp engine/human_limit/nnue_features.cpp engine/human_limit/accumulator.cpp engine/human_limit/search.cpp
ANCIENT_ENGINE_SRC = engine/ancient_engine/evaluation.cpp engine/ancient_engine/search.cpp
OLD_ENGINE_SRC = engine/old_engine/evaluation.cpp engine/old_engine/search.cpp

engine_gameplay: engine_gameplay.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC) chess/board.h chess/pieces.h engine/human_limit/search.h engine/human_limit/network.h
	$(CXX) $(CXXFLAGS) -I. -o engine_gameplay engine_gameplay.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC)

engine_selfplay: engine_selfplay.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC) chess/board.h chess/pieces.h engine/human_limit/search.h engine/human_limit/network.h
	$(CXX) $(CXXFLAGS) -I. -o engine_selfplay engine_selfplay.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC)

bestmove_cli: tools/bestmove_cli.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC) $(ANCIENT_ENGINE_SRC) $(OLD_ENGINE_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tools/bestmove_cli tools/bestmove_cli.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC) $(ANCIENT_ENGINE_SRC) $(OLD_ENGINE_SRC)

# Persistent UCI front-end for human_limit (keeps TT/history across moves, supports
# Lazy SMP via the Threads option). Needs threads + a larger per-thread stack for deep recursion.
uci_engine: tools/uci_engine.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC) engine/human_limit/search.h engine/human_limit/network.h
	$(CXX) $(CXXFLAGS) -pthread -Wl,--stack,33554432 -I. -o tools/uci_engine tools/uci_engine.cpp $(CHESS_SRC) $(BITBOARD_SRC) $(HUMAN_LIMIT_SRC)

eval_cli: tools/eval_cli.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tools/eval_cli tools/eval_cli.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC)

test: tests/test_chess.cpp tests/simple_demo.cpp tests/test_nnue_features.cpp tests/test_accumulator.cpp tests/test_fast_legality.cpp tests/test_bitboard_attacks.cpp tests/test_bitboard_perft.cpp tests/test_bitboard_cross_validate.cpp tests/test_accumulator_bb.cpp $(CHESS_SRC) $(BITBOARD_SRC) chess/board.h chess/pieces.h engine/human_limit/nnue_features.h engine/human_limit/accumulator.h
	$(CXX) $(CXXFLAGS) -o tests/test_chess tests/test_chess.cpp $(CHESS_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/simple_demo tests/simple_demo.cpp $(CHESS_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/test_nnue_features tests/test_nnue_features.cpp $(CHESS_SRC) engine/human_limit/nnue_features.cpp
	$(CXX) $(CXXFLAGS) -I. -o tests/test_accumulator tests/test_accumulator.cpp $(CHESS_SRC) engine/human_limit/network.cpp engine/human_limit/nnue_features.cpp engine/human_limit/accumulator.cpp
	$(CXX) $(CXXFLAGS) -o tests/test_fast_legality tests/test_fast_legality.cpp $(CHESS_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/test_bitboard_attacks tests/test_bitboard_attacks.cpp chess/bitboard/bitboard.cpp chess/bitboard/magic.cpp
	$(CXX) $(CXXFLAGS) -I. -o tests/test_bitboard_perft tests/test_bitboard_perft.cpp $(BITBOARD_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/test_bitboard_cross_validate tests/test_bitboard_cross_validate.cpp $(BITBOARD_SRC) $(CHESS_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/test_accumulator_bb tests/test_accumulator_bb.cpp $(BITBOARD_SRC) $(CHESS_SRC) engine/human_limit/network.cpp engine/human_limit/nnue_features.cpp engine/human_limit/accumulator.cpp
	./tests/test_chess
	./tests/test_nnue_features
	./tests/test_accumulator
	./tests/test_fast_legality
	./tests/test_bitboard_attacks
	./tests/test_bitboard_perft
	./tests/test_bitboard_cross_validate
	./tests/test_accumulator_bb

all: engine_gameplay engine_selfplay uci_engine test

clean:
	rm -f engine_gameplay engine_selfplay train_human_limit tools/bestmove_cli tools/uci_engine tools/eval_cli tests/test_chess tests/simple_demo tests/test_nnue_features tests/test_accumulator tests/test_fast_legality tests/test_bitboard_attacks tests/test_bitboard_perft tests/test_bitboard_cross_validate tests/test_accumulator_bb

.PHONY: all test clean
