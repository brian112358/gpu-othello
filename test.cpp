#include <cstdio>
#include <cassert>

#include "board.hpp"
#include "common.hpp"
#include "player.hpp"

int main(int argc, char *argv[]) {
    Player *player[2];
    player[0] = new Player(BLACK);
    player[1] = new Player(WHITE);
    Board b;
    Move *m[2];
    m[0] = nullptr;
    m[1] = nullptr;
	int idx = 0;
    while (!b.isDone()) {
		m[idx] = player[idx]->doMove(m[(idx+1)%2], 300000);
		// b.printBoard();
		if (m[idx] && *m[idx] != Move(-1, -1)) {
            assert(b.doMove(*m[idx], idx? WHITE:BLACK));
        }
		idx = (idx+1)%2;
		if (m[idx]) {
			delete m[idx];
			m[idx] = nullptr;
		}
    }
    delete player[0];
    delete player[1];
    if (m[(idx+1)%2]) {
    	delete m[(idx+1)%2];
    }
}