#ifndef __PLAYER_H__
#define __PLAYER_H__

#include "common.hpp"
#include "board.hpp"

#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

class Player {

public:
    Player(Side side);
    ~Player();
    
    Move *doMove(Move *opponentsMove, int msLeft);
    Board *board; // Board used to store Othello game state

private:
    Side side;  // Stores the side (BLACK/WHITE) the player is on
};

#endif
