CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -flto

CHESS_SRC = chess/board.cpp chess/pieces.cpp
HUMAN_LIMIT_SRC = engine/human_limit/network.cpp engine/human_limit/search.cpp
ANCIENT_ENGINE_SRC = engine/ancient_engine/evaluation.cpp engine/ancient_engine/search.cpp
OLD_ENGINE_SRC = engine/old_engine/evaluation.cpp engine/old_engine/search.cpp

engine_gameplay: engine_gameplay.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC) chess/board.h chess/pieces.h engine/human_limit/search.h engine/human_limit/network.h
	$(CXX) $(CXXFLAGS) -I. -o engine_gameplay engine_gameplay.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC)

engine_selfplay: engine_selfplay.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC) chess/board.h chess/pieces.h engine/human_limit/search.h engine/human_limit/network.h
	$(CXX) $(CXXFLAGS) -I. -o engine_selfplay engine_selfplay.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC)

train_human_limit: train_human_limit.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC) chess/board.h chess/pieces.h engine/human_limit/search.h engine/human_limit/network.h
	$(CXX) $(CXXFLAGS) -I. -o train_human_limit train_human_limit.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC)

bestmove_cli: tools/bestmove_cli.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC) $(ANCIENT_ENGINE_SRC) $(OLD_ENGINE_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tools/bestmove_cli tools/bestmove_cli.cpp $(CHESS_SRC) $(HUMAN_LIMIT_SRC) $(ANCIENT_ENGINE_SRC) $(OLD_ENGINE_SRC)

test: tests/test_chess.cpp tests/simple_demo.cpp $(CHESS_SRC) chess/board.h chess/pieces.h
	$(CXX) $(CXXFLAGS) -o tests/test_chess tests/test_chess.cpp $(CHESS_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/simple_demo tests/simple_demo.cpp $(CHESS_SRC)
	./tests/test_chess

all: engine_gameplay engine_selfplay test

clean:
	rm -f engine_gameplay engine_selfplay train_human_limit tools/bestmove_cli tests/test_chess tests/simple_demo

.PHONY: all test clean
