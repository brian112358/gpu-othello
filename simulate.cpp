#include "simulate.hpp"

#include "board.hpp"

void simulateNode(Node *n, int numSims) {
	int numWins = 0;
	n->updateSim(numSims, numWins);
}