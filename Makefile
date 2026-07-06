CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

CHESS_SRC = chess/board.cpp chess/pieces.cpp
ENGINE_SRC = engine/evaluation.cpp engine/search.cpp

engine_gameplay: engine_gameplay.cpp $(CHESS_SRC) $(ENGINE_SRC) chess/board.h chess/pieces.h engine/search.h engine/evaluation.h
	$(CXX) $(CXXFLAGS) -I. -o engine_gameplay engine_gameplay.cpp $(CHESS_SRC) $(ENGINE_SRC)

engine_selfplay: engine_selfplay.cpp $(CHESS_SRC) $(ENGINE_SRC) chess/board.h chess/pieces.h engine/search.h engine/evaluation.h
	$(CXX) $(CXXFLAGS) -I. -o engine_selfplay engine_selfplay.cpp $(CHESS_SRC) $(ENGINE_SRC)

test: tests/test_chess.cpp tests/simple_demo.cpp $(CHESS_SRC) chess/board.h chess/pieces.h
	$(CXX) $(CXXFLAGS) -o tests/test_chess tests/test_chess.cpp $(CHESS_SRC)
	$(CXX) $(CXXFLAGS) -I. -o tests/simple_demo tests/simple_demo.cpp $(CHESS_SRC)
	./tests/test_chess

all: engine_gameplay engine_selfplay test

clean:
	rm -f engine_gameplay engine_selfplay tests/test_chess tests/simple_demo

.PHONY: all test clean
