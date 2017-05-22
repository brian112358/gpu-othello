#include "player.hpp"

#include <iostream>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cstdlib>

/*
 * Constructor for the player; initialize everything here. The side your AI is
 * on (BLACK or WHITE) is passed in as "side". The constructor must finish 
 * within 30 seconds.
 */
Player::Player(Side side) {    
    board = new Board();
    this->side = side;
}

/*
 * Destructor for the player.
 */
Player::~Player() {
    delete board;
}

/*
 * Compute the next move given the opponent's last move. Your AI is
 * expected to keep track of the board on its own. If this is the first move,
 * or if the opponent passed on the last move, then opponentsMove will be nullptr.
 *
 * msLeft represents the time your AI has left for the total game, in
 * milliseconds. doMove() must take no longer than msLeft, or your AI will
 * be disqualified! An msLeft value of -1 indicates no time limit.
 *
 * The move returned must be legal; if there are no valid moves for your side,
 * return nullptr.
 */
Move *Player::doMove(Move *opponentsMove, int msLeft) {
    if (opponentsMove) {
        fprintf(stderr, "Opponent move: (%d, %d)\n", opponentsMove->x, opponentsMove->y);
        board->doMove(*opponentsMove, OTHER(side));
    }

    board->printBoard();

    Move *moves = new Move[64];

    int numMoves = board->getMovesAsArray(moves, side);

    if (!numMoves) return nullptr;

    Move *move = new Move(moves[rand() % numMoves]);
    delete[] moves;

    board->doMove(*move, side);

    fprintf(stderr, "My move: (%d, %d)\n", move->x, move->y);
    board->printBoard();

    return move;
}
