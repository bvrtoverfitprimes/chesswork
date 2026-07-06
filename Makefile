CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

CHESS_SRC = chess/board.cpp chess/pieces.cpp
ENGINE_SRC = engine/evaluation.cpp engine/search.cpp

simple_demo: simple_demo.cpp $(CHESS_SRC) chess/board.h chess/pieces.h
	$(CXX) $(CXXFLAGS) -o simple_demo simple_demo.cpp $(CHESS_SRC)

engine_gameplay: engine/engine_gameplay.cpp $(CHESS_SRC) $(ENGINE_SRC) chess/board.h chess/pieces.h engine/search.h engine/evaluation.h
	$(CXX) $(CXXFLAGS) -o engine_gameplay engine/engine_gameplay.cpp $(CHESS_SRC) $(ENGINE_SRC)

engine_selfplay: engine/engine_selfplay.cpp $(CHESS_SRC) $(ENGINE_SRC) chess/board.h chess/pieces.h engine/search.h engine/evaluation.h
	$(CXX) $(CXXFLAGS) -o engine_selfplay engine/engine_selfplay.cpp $(CHESS_SRC) $(ENGINE_SRC)

test: tests/test_chess.cpp $(CHESS_SRC) chess/board.h chess/pieces.h
	$(CXX) $(CXXFLAGS) -o tests/test_chess tests/test_chess.cpp $(CHESS_SRC)
	./tests/test_chess

all: simple_demo engine_gameplay engine_selfplay

clean:
	rm -f simple_demo engine_gameplay engine_selfplay tests/test_chess

.PHONY: all test clean
