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

// Start using minimax at this turn
#define MINIMAX_TURN 0

const float time_alloc[60] =
{
    0.0050, 0.0096, 0.0148, 0.0148, 0.0148, 0.0167,
    0.0167, 0.0176, 0.0177, 0.0177, 0.0177, 0.0177,
    0.0206, 0.0212, 0.0216, 0.0216, 0.0235, 0.0245,
    0.0245, 0.0245, 0.0274, 0.0280, 0.0286, 0.0304,
    0.0313, 0.0313, 0.0343, 0.0352, 0.0374, 0.0382,
    0.0411, 0.0421, 0.0443, 0.0450, 0.0479, 0.0518,
    0.0547, 0.0582, 0.0623, 0.0655, 0.0715, 0.0746,
    0.0798, 0.0864, 0.0953, 0.1049, 0.1156, 0.1283,
    0.1451, 0.1590, 0.1805, 0.2015, 0.2127, 0.2353,
    0.2681, 0.3067, 0.3667, 0.4610, 0.6319, 0.9005
};

// const float time_alloc[60] =
// {
//     0.0153, 0.0201, 0.0241, 0.0200, 0.0173, 0.0173,
//     0.0156, 0.0151, 0.0140, 0.0131, 0.0123, 0.0116,
//     0.0129, 0.0127, 0.0124, 0.0120, 0.0126, 0.0127,
//     0.0123, 0.0120, 0.0131, 0.0131, 0.0131, 0.0136,
//     0.0138, 0.0136, 0.0146, 0.0148, 0.0155, 0.0157,
//     0.0167, 0.0170, 0.0177, 0.0179, 0.0189, 0.0203,
//     0.0214, 0.0227, 0.0243, 0.0255, 0.0279, 0.0292,
//     0.0313, 0.0341, 0.0379, 0.0420, 0.0469, 0.0529,
//     0.0609, 0.0683, 0.0796, 0.0919, 0.1009, 0.1167,
//     0.1401, 0.1710, 0.2214, 0.3090, 0.4917, 0.9000
// };

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
    int timeBudgetMs = msLeft < 0? 4000:(time_alloc[moveNumber] * msLeft);
    if (timeBudgetMs < 500) timeBudgetMs = 500;
    
    fprintf(stderr, "Allocated %d of %d ms on move %d\n",
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

    Move *move;
    Move moves[MAX_NUM_MOVES];
    int numMoves = board->getMovesAsArray(moves, this->side);

    // Early exits:
    if (numMoves == 0) {
        fprintf(stderr, "[No moves]\n");
        return nullptr;
    }
    else if (numMoves == 1) {
        move = new Move(moves[0]);
        board->doMove(*move, this->side);
        fprintf(stderr, "[1 move]: (%d, %d)\n", move->x, move->y);
        return move;
    }
    else if (root->state & PROVEN_WIN) {
        move = new Move();
        root->getBestMove(move, true, true);
        board->doMove(*move, this->side);
    }

    bool useMinimax = moveNumber > MINIMAX_TURN;

    #ifdef GPU_ON
        // expandGameTreeGpu(root, useMinimax, timeBudgetMs);
        expandGameTreeGpuBlock(root, useMinimax, timeBudgetMs);
    #else
        expandGameTree(root, useMinimax, timeBudgetMs);
    #endif

    move = new Move();
    int numRetries = 0;
    while (!root->getBestMove(move, useMinimax, numRetries > 5)) {
        #ifdef GPU_ON
            // expandGameTreeGpu(root, useMinimax, timeBudgetMs/10);
            expandGameTreeGpuBlock(root, useMinimax, timeBudgetMs/10);
        #else
            expandGameTree(root, useMinimax, timeBudgetMs/10);
        #endif
        numRetries++;
    }

    fprintf(stderr, "Game tree now has %d nodes\n", root->numDescendants);

    board->doMove(*move, this->side);

    return move;
}
