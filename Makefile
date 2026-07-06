CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

simple_demo: simple_demo.cpp chess/board.cpp chess/pieces.cpp chess/board.h chess/pieces.h
	$(CXX) $(CXXFLAGS) -o simple_demo simple_demo.cpp chess/board.cpp chess/pieces.cpp

test: tests/test_chess.cpp chess/board.cpp chess/pieces.cpp chess/board.h chess/pieces.h
	$(CXX) $(CXXFLAGS) -o tests/test_chess tests/test_chess.cpp chess/board.cpp chess/pieces.cpp
	./tests/test_chess

clean:
	rm -f simple_demo tests/test_chess

.PHONY: test clean
