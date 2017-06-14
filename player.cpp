#include "player.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "gametree.hpp"
#include "simulate.hpp"

// Comment out to use CPU only
#define GPU_ON

// Comment out to only do a rollout from a single child at a time.
#define BLOCK_PARALLEL

// Use minimax for backpropogation of scores?
#define USE_MINIMAX true

const float time_alloc[60] =
{
    0.0123, 0.0151, 0.0231, 0.0231, 0.0228, 0.0258,
    0.0255, 0.0268, 0.0267, 0.0266, 0.0263, 0.0263,
    0.0303, 0.0311, 0.0313, 0.0312, 0.0336, 0.0350,
    0.0345, 0.0345, 0.0381, 0.0388, 0.0392, 0.0415,
    0.0423, 0.0421, 0.0455, 0.0467, 0.0489, 0.0497,
    0.0529, 0.0539, 0.0561, 0.0567, 0.0596, 0.0641,
    0.0668, 0.0708, 0.0747, 0.0781, 0.0841, 0.0873,
    0.0919, 0.0991, 0.1075, 0.1178, 0.1276, 0.1412,
    0.1567, 0.1714, 0.1906, 0.2124, 0.2187, 0.2417,
    0.2670, 0.3055, 0.3507, 0.4417, 0.9000, 0.9000
};

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
    if (root) delete root;
}

Move *Player::doMove(Move *opponentsMove, int msLeft) {
    if (opponentsMove) {
        board->doMove(*opponentsMove, OTHER(this->side));
    }

    const int moveNumber = board->countPieces() - 4;
    assert(0 <= moveNumber && moveNumber <= 60);
    int timeBudgetMs = msLeft < 0? 8000:(time_alloc[moveNumber] * msLeft);
    if (timeBudgetMs < 500) timeBudgetMs = 500;
    
    fprintf(stderr, "\nAllocated %d of %d ms on move %d.\n",
        timeBudgetMs, msLeft, moveNumber);

    // If we have previous game tree info
    if (root) {
        Node *new_root = root->searchBoard(*board, this->side, 2);
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
            assert (root->side == this->side);
            fprintf(stderr, "Saved %u old simulations in %u nodes.\n", root->numSims, root->numDescendants);
        }
        else {
            delete root;
            root = new Node(*board, nullptr, this->side);
        }
    }
    else {
        root = new Node(*board, nullptr, this->side);
    }

    Move *move;
    Move moves[MAX_NUM_MOVES];
    int numMoves = board->getMovesAsArray(moves, this->side);

    // Early exits:
    if (numMoves == 0) {
        fprintf(stderr, "[No moves]: PASS\n\n");
        return nullptr;
    }
    else if ((root->state & SCORE_FINAL) && !(root->state & PROVEN_LOSS)) {
        move = new Move();
        root->getBestMove(move, true, true);
        board->doMove(*move, this->side);
        fprintf(stderr, "Game tree now has %d nodes\n\n", root->numDescendants);
        return move;
    }
    else if (numMoves == 1) {
        move = new Move(moves[0]);
        board->doMove(*move, this->side);
        fprintf(stderr, "[1 move]: (%d, %d)\n\n", move->x, move->y);
        return move;
    }

    int numGpuSimNodes = 0;

    #ifdef GPU_ON
        #ifdef BLOCK_PARALLEL
            numGpuSimNodes = expandGameTreeGpuBlock(root, USE_MINIMAX, timeBudgetMs);
        #else
            numGpuSimNodes = expandGameTreeGpu(root, USE_MINIMAX, timeBudgetMs);
        #endif
    #else
        expandGameTree(root, USE_MINIMAX, timeBudgetMs);
    #endif

    fprintf(stderr, "%d nodes simulated with GPU\n", numGpuSimNodes);

    move = new Move();
    int numRetries = 0;
    while (!root->getBestMove(move, USE_MINIMAX, numRetries > 9)) {
        fprintf(stderr, ".");
        #ifdef GPU_ON
            #ifdef BLOCK_PARALLEL
                numGpuSimNodes += expandGameTreeGpuBlock(root, USE_MINIMAX, timeBudgetMs/10);
            #else
                numGpuSimNodes += expandGameTreeGpu(root, USE_MINIMAX, timeBudgetMs/10);
            #endif
        #else
            expandGameTree(root, USE_MINIMAX, timeBudgetMs/10);
        #endif
        numRetries++;
    }

    fprintf(stderr, "Game tree now has %d nodes\n\n", root->numDescendants);

    board->doMove(*move, this->side);

    return move;
}
