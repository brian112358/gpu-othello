#include "player.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "gametree.hpp"
#include "simulate.hpp"

/*
 * Constructor for the player; initialize everything here. The side your AI is
 * on (BLACK or WHITE) is passed in as "side". The constructor must finish 
 * within 30 seconds.
 */
Player::Player(Side side) {    
    board = new Board();
    this->side = side;
    srand(time(NULL));
}

/*
 * Destructor for the player.
 */
Player::~Player() {
    delete board;
}

#define NUM_ITERS 100000
#define SIMS_PER_ITER 10

Move *Player::doMove(Move *opponentsMove, int msLeft) {
    fprintf(stderr, "msLeft: %d\n", msLeft);
    if (opponentsMove) {
        // fprintf(stderr, "Opponent move: (%d, %d)\n", opponentsMove->x, opponentsMove->y);
        board->doMove(*opponentsMove, OTHER(this->side));
    }

    // board->printBoard();
    
    Node root(*board, nullptr, this->side);
    simulateNode(&root, 1);

    for (int i = 0; i < NUM_ITERS; i++) {
        Node *expandedNode = root.searchScore();
        simulateNode(expandedNode, SIMS_PER_ITER);
    }

    Move *move = new Move();
    while (!root.getBestMove(move)) {
        for (int i = 0; i < NUM_ITERS/10; i++) {
            Node *expandedNode = root.searchScore();
            simulateNode(expandedNode, SIMS_PER_ITER);
        }
    }

    board->doMove(*move, this->side);

    // fprintf(stderr, "My move: (%d, %d)\n", move->x, move->y);

    return move;
}
