#ifndef __BOARD_HPP__
#define __BOARD_HPP__

#include <cstdint>
#include "common.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

// Defines a bitmap corresponding to the 8x8 Othello board. Spaces are defined
// row-major, with index increasing to the right and down.
typedef uint64_t bitboard;

__host__ __device__ int countSparse(bitboard b);
__host__ __device__ int count(bitboard b);
void print(bitboard b);

class Board {
  public:
    __host__ __device__ Board();
    __host__ __device__ ~Board();
    __host__ __device__ bitboard getOccupied();
    __host__ __device__ bitboard getEmpty();
    __host__ __device__ bitboard getMoves(Side side);
    __host__ __device__ int getMovesAsArray(Move *output_moves, Side side);
    __host__ __device__ int getHeuristicMovesAsArray(Move *output_moves, Side side);


    __host__ __device__ bool get(Side side, int x, int y);
    __host__ __device__ bool isDone();
    __host__ __device__ bool hasMoves(Side side);
    __host__ __device__ int numMoves(Side side);
    __host__ __device__ bool checkMove(Move m, Side side);
    __host__ __device__ bool doMove(Move m, Side side);
    __host__ __device__ bool doMove(int m, Side side);
    __host__ __device__ int countPieces(Side side);
    __host__ __device__ int countPieces();
    __host__ __device__ int countEmpty();

    // // Heuristics
    __host__ __device__ float getHeuristic(Side side);
    __host__ __device__ float getCornersHeuristic(Side side);
    __host__ __device__ float getXSquaresHeuristic(Side side);
    __host__ __device__ float getCSquaresHeuristic(Side side);
    __host__ __device__ float getMobilityHeuristic(Side side);
    __host__ __device__ float getParityHeuristic(Side side);
    __host__ __device__ float getFrontierHeuristic(Side side);
    __host__ __device__ float getPiecesHeuristic(Side side);

    __host__ __device__ bool operator==(const Board &other) const;
    __host__ __device__ bool operator!=(const Board &other) const;
    void printBoard();

  private:
    bitboard occupied[2]; // occupied[Side] represents pieces for each Side
};

__host__ __device__ bitboard allSandwiched(bitboard gen1, bitboard gen2, bitboard prop);
__host__ __device__ bitboard allAttack(bitboard gen, bitboard prop);
__host__ __device__ bitboard allShift(bitboard gen);
__host__ __device__ bitboard diagShift(bitboard gen);
__host__ __device__ bitboard cardShift(bitboard gen);

__host__ __device__ bitboard   SFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard   NFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard   EFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard   WFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard  NEFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard  SEFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard  SWFill(bitboard gen, bitboard prop);
__host__ __device__ bitboard  NWFill(bitboard gen, bitboard prop);

__host__ __device__ bitboard  SShift (bitboard b);
__host__ __device__ bitboard  NShift (bitboard b);
__host__ __device__ bitboard  EShift (bitboard b);
__host__ __device__ bitboard NEShift (bitboard b);
__host__ __device__ bitboard SEShift (bitboard b);
__host__ __device__ bitboard  WShift (bitboard b);
__host__ __device__ bitboard SWShift (bitboard b);
__host__ __device__ bitboard NWShift (bitboard b);

#endif
