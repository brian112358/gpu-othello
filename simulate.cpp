#include "simulate.hpp"

#include <cstdio>
#include <cstdlib>
#include "board.hpp"
#include "common.hpp"

void simulateNode(Node *n, int numSims) {
    int winDiff = 0;
    for (int i = 0; i < numSims; i++) {
        Board b = n->board;
        Move *moves = new Move[60];
        Side side = n->side;
        int numMoves;
        while (!b.isDone()) {
            numMoves = b.getMovesAsArray(moves, side);
            if (numMoves) {
                b.doMove(Move(moves[rand() % numMoves]), side);
            }
            side = OTHER(side);
        }
        // fprintf(stderr, "%d Black, %d White\n", b.countPieces(BLACK), b.countPieces(WHITE));
        if (b.countPieces(n->side) > b.countPieces(OTHER(n->side))) {
            winDiff++;
        }
        else {
            winDiff--;
        }
        delete[] moves;
    }
    n->updateSim(numSims, winDiff);
}
