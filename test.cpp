#include <cstdio>

#include "board.hpp"
#include "common.hpp"

int main(int argc, char *argv[]) {
	Board b = Board();
	b.printBoard();
	printf("Black: %d\tWhite: %d\tTotal: %d\n",
		b.countPieces(BLACK), b.countPieces(WHITE), b.countPieces());
	printf("Black's moves:\n");
	print(b.getMoves(BLACK));


    Move *moves = new Move[60];
    int num_moves = b.getMovesAsArray(moves, BLACK);

    for (int i = 0; i < num_moves; i++) {
    	printf("(%d, %d), ", moves[i].x, moves[i].y);
    }
    printf("\n");

	printf("Move completed: %s\n", b.doMove(Move(5, 4), BLACK)? "True":"False");


	b.printBoard();
	printf("Black: %d\tWhite: %d\tTotal: %d\n",
		b.countPieces(BLACK), b.countPieces(WHITE), b.countPieces());

	printf("White's moves:\n");
	print(b.getMoves(WHITE));
 }