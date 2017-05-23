#include <cstdio>

#include "board.hpp"
#include "common.hpp"
#include "player.hpp"

int main(int argc, char *argv[]) {
    Player *player = new Player(BLACK);
    delete player->doMove(nullptr, -1);
    delete player;
}