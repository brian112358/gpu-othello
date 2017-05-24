#include "player.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "gametree.hpp"
#include "simulate.hpp"

// Comment out to use CPU only
#define GPU_ON

/*
 * Constructor for the player; initialize everything here. The side your AI is
 * on (BLACK or WHITE) is passed in as "side". The constructor must finish 
 * within 30 seconds.
 */
Player::Player(Side side) {    
    board = new Board();
    root = nullptr;
    this->side = side;
    srand(time(NULL));
}

/*
 * Destructor for the player.
 */
Player::~Player() {
    delete board;
}

// #define NUM_ITERS 10000
// #define CPU_SIMS_PER_ITER 1
// #define GPU_SIMS_PER_ITER 128

Move *Player::doMove(Move *opponentsMove, int msLeft) {
    fprintf(stderr, "msLeft: %d\n", msLeft);
    if (opponentsMove) {
        // fprintf(stderr, "Opponent move: (%d, %d)\n", opponentsMove->x, opponentsMove->y);
        board->doMove(*opponentsMove, OTHER(this->side));
    }

    // int timeBudgetMs = msLeft / 60;
    // int movesLeft = board->countEmpty();
    int timeBudgetMs = 4000;

    // board->printBoard();
    
    // If we have previous game tree info
    if (root) {
        Node *new_root = root->searchBoard(*board, 2);
        if (new_root) {
            // Remove new_root from the old game tree
            for (uint i = 0; i < new_root->parent->children.size(); i++) {
                if (new_root->parent->children[i] == new_root) {
                    new_root->parent->children[i] = nullptr;
                }
            }
            new_root->parent = nullptr;
            delete root;
            root = new_root;
            fprintf(stderr, "Saved %d old simulations\n", root->numSims);
        }
        else {
            delete root;
            root = new Node(*board, nullptr, this->side);
        }
    }
    else {
        root = new Node(*board, nullptr, this->side);
    }
    
    #ifdef GPU_ON
        fprintf(stderr, "Ran %d iterations\n", expandGameTreeGpu(*root, timeBudgetMs));
    #else
        expandGameTree(*root, timeBudgetMs);
    #endif

    Move *move = new Move();
    while (!root->getBestMove(move)) {
            #ifdef GPU_ON
            expandGameTreeGpu(*root, timeBudgetMs/10);
        #else
            expandGameTree(*root, timeBudgetMs/10);
        #endif
    }

    board->doMove(*move, this->side);


    // fprintf(stderr, "My move: (%d, %d)\n", move->x, move->y);

    return move;
}
