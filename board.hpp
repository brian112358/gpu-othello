#ifndef __BOARD_HPP__
#define __BOARD_HPP__

#include <cstdint>
#include "common.hpp"

// Defines a bitmap corresponding to the 8x8 Othello board. Spaces are defined
// row-major, with index increasing to the right and down.
typedef uint64_t bitboard;

int countSparse(bitboard b);
int count(bitboard b);
void print(bitboard b);

class Board {
  public:
    Board();
    ~Board();
    bitboard getOccupied();
    bitboard getEmpty();
    bitboard getMoves(Side side);
    int getMovesAsArray(Move *output_moves, Side side);


    bool get(Side side, int x, int y);
    bool isDone();
    bool hasMoves(Side side);
    int numMoves(Side side);
    bool checkMove(Move m, Side side);
    bool doMove(Move m, Side side);
    int countPieces(Side side);
    int countPieces();
    void printBoard();

  private:
    bitboard occupied[2]; // occupied[Side] represents pieces for each Side
};

bitboard allSandwiched(bitboard gen1, bitboard gen2, bitboard prop);
bitboard allAttack(bitboard gen, bitboard prop);

bitboard   SFill(bitboard gen, bitboard prop);
bitboard   NFill(bitboard gen, bitboard prop);
bitboard   EFill(bitboard gen, bitboard prop);
bitboard   WFill(bitboard gen, bitboard prop);
bitboard  NEFill(bitboard gen, bitboard prop);
bitboard  SEFill(bitboard gen, bitboard prop);
bitboard  SWFill(bitboard gen, bitboard prop);
bitboard  NWFill(bitboard gen, bitboard prop);

bitboard  SShift (bitboard b);
bitboard  NShift (bitboard b);
bitboard  EShift (bitboard b);
bitboard NEShift (bitboard b);
bitboard SEShift (bitboard b);
bitboard  WShift (bitboard b);
bitboard SWShift (bitboard b);
bitboard NWShift (bitboard b);

#endif
