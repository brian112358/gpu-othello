#include <cstdio>
#include "board.hpp"

__host__ __device__ int xyToIndex(int x, int y) { return 8 * y + x; }
__host__ __device__ int moveToIndex(Move m) { return xyToIndex(m.x, m.y); }
__host__ __device__ Move indexToMove(int i) { return Move(i % 8, i / 8); }
__host__ __device__ bitboard indexToBitboard(int i) { return 1ULL << i; }
__host__ __device__ bitboard xyToBitboard(int x, int y) { return indexToBitboard(xyToIndex(x,y)); }
__host__ __device__ bitboard moveToBitboard(Move m) { return indexToBitboard(moveToIndex(m)); }

/*
 * Make a standard 8x8 Othello board and initialize it to the standard setup.
 */
__host__ __device__
Board::Board() {
    occupied[WHITE] = 0ULL;
    occupied[WHITE] |= xyToBitboard(3, 3);
    occupied[WHITE] |= xyToBitboard(4, 4);
    occupied[BLACK] = 0ULL;
    occupied[BLACK] |= xyToBitboard(3, 4);
    occupied[BLACK] |= xyToBitboard(4, 3);
}

/*
 * Destructor for the board.
 */
__host__ __device__
Board::~Board() {
}

// Assume that the board is sparsely populated
__host__ __device__
int countSparse(bitboard b) {
    int count = 0;
    while (b) {
        count++;
        b &= b - 1; // reset LS1B
    }
    return count;
}

// Faster if board is highly populated
// http://chessprogramming.wikispaces.com/Population+Count
#define k1 0x5555555555555555ULL
#define k2 0x3333333333333333ULL
#define k4 0x0f0f0f0f0f0f0f0fULL
#define kf 0x0101010101010101ULL

__host__ __device__
int count(bitboard b) {
    b =  b       - ((b >> 1)  & k1); /* put count of each 2 bits into those 2 bits */
    b = (b & k2) + ((b >> 2)  & k2); /* put count of each 4 bits into those 4 bits */
    b = (b       +  (b >> 4)) & k4 ; /* put count of each 8 bits into those 8 bits */
    b = (b * kf) >> 56; /* returns 8 most significant bits of b + (b<<8) + (b<<16) + (b<<24) + ...  */
    return (int) b;
}

void print(bitboard b) {
    fprintf(stderr, "  01234567\n");
    for (int y = 0; y < 8; y++) {
        fprintf(stderr, "%d ", y);
        for (int x = 0; x < 8; x++) {
            if (b & xyToBitboard(x, y)) {
                fprintf(stderr, "*");
            }
            else {
                fprintf(stderr, "-");
            }
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}

__host__ __device__
bitboard Board::getOccupied() {
    return occupied[WHITE] | occupied[BLACK];
}

__host__ __device__
bitboard Board::getEmpty() {
    return ~getOccupied();
}

__host__ __device__
bitboard Board::getMoves(Side side) {
    return allAttack(occupied[side], occupied[!side]) & getEmpty();
}

__host__ __device__
int Board::getMovesAsArray(Move *output_moves, Side side) {
    bitboard moves = getMoves(side);
    if (!moves) return 0;
    int numMoves = 0;
    do {
        int idx;
        #ifdef  __CUDA_ARCH__
          idx = __ffsll(moves);
        #else
          idx = __builtin_ffsll(moves);
        #endif
        output_moves[numMoves] = indexToMove(idx-1);
        numMoves++;
    } while (moves &= moves-1); // reset LS1B
    return numMoves;
}

__host__ __device__
bool Board::get(Side side, int x, int y) {
    return occupied[side] & xyToBitboard(x, y);
}

__host__ __device__
bool Board::isDone() {
    return !(hasMoves(WHITE) || hasMoves(BLACK));
}

__host__ __device__
bool Board::hasMoves(Side side) {
    return getMoves(side);
}

__host__ __device__
int Board::numMoves(Side side) {
    return countSparse(getMoves(side));
}

__host__ __device__
bool Board::checkMove(Move m, Side side) {
    return getMoves(side) & moveToBitboard(m);
}

__host__ __device__
bool Board::doMove(Move m, Side side) {
    // Return false if move is not valid
    if (!(getMoves(side) & moveToBitboard(m))) {
        return false;
    }
    bitboard move = moveToBitboard(m);
    bitboard flipped = allSandwiched(move, occupied[side], occupied[!side]);

    occupied[side] |= move;
    occupied[side] |= flipped;
    occupied[!side] ^= flipped;
    return true;
}

__host__ __device__
int Board::countPieces(Side side) {
    return count(occupied[side]);
}

__host__ __device__
int Board::countPieces() {
    return count(getOccupied());
}

__host__ __device__
int Board::countEmpty() {
    return count(getEmpty());
}

__host__ __device__ bool Board::operator==(const Board &other) const {
    return this->occupied[0] == other.occupied[0] &&
        this->occupied[1] == other.occupied[1];
}
__host__ __device__ bool Board::operator!=(const Board &other) const {
    return !(*this == other);
}

void Board::printBoard() {
    fprintf(stderr, "  01234567\n");
    for (int y = 0; y < 8; y++) {
        fprintf(stderr, "%d ", y);
        for (int x = 0; x < 8; x++) {
            if (occupied[BLACK] & xyToBitboard(x, y)) {
                fprintf(stderr, "B");
            }
            else if (occupied[WHITE] & xyToBitboard(x, y)) {
                fprintf(stderr, "W");
            }
            else {
                fprintf(stderr, "-");
            }
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "\n");
}


// Bitboard algorithms

/**
 *   NW           N          NE
 *         -9    -8    -7
 *             \  |  /
 *    W    -1 <-  0 -> +1    E
 *             /  |  \
 *         +7    +8    +9
 *    S           S          SE
 */

// Bitmasks to prevent wrapping around the E and W directions
#define notE 0xFEFEFEFEFEFEFEFEULL
#define notW 0x7F7F7F7F7F7F7F7FULL

__host__ __device__
bitboard allSandwiched(bitboard gen1, bitboard gen2, bitboard prop) {
    bitboard flood =  SFill(gen1, prop) &  NFill(gen2, prop);
    flood         |=  NFill(gen1, prop) &  SFill(gen2, prop);
    flood         |=  EFill(gen1, prop) &  WFill(gen2, prop);
    flood         |= SEFill(gen1, prop) & NWFill(gen2, prop);
    flood         |= NEFill(gen1, prop) & SWFill(gen2, prop);
    flood         |=  WFill(gen1, prop) &  EFill(gen2, prop);
    flood         |= SWFill(gen1, prop) & NEFill(gen2, prop);
    flood         |= NWFill(gen1, prop) & SEFill(gen2, prop);
    return flood;
}

__host__ __device__
bitboard allAttack(bitboard gen, bitboard prop) {
    bitboard flood =  SShift( SFill(gen, prop));
    flood         |=  NShift( NFill(gen, prop));
    flood         |=  EShift( EFill(gen, prop));
    flood         |= SEShift(SEFill(gen, prop));
    flood         |= NEShift(NEFill(gen, prop));
    flood         |=  WShift( WFill(gen, prop));
    flood         |= SWShift(SWFill(gen, prop));
    flood         |= NWShift(NWFill(gen, prop));
    return flood;
}

// Dumb7Fill: http://chessprogramming.wikispaces.com/Dumb7Fill
__host__ __device__
bitboard  SFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    flood |= gen = (gen   << 8) & prop;
    flood |= gen = (gen   << 8) & prop;
    flood |= gen = (gen   << 8) & prop;
    flood |= gen = (gen   << 8) & prop;
    flood |= gen = (gen   << 8) & prop;
    flood |=       (gen   << 8) & prop;
    return flood;
}

__host__ __device__
bitboard  NFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    flood |= gen = (gen   >> 8) & prop;
    flood |= gen = (gen   >> 8) & prop;
    flood |= gen = (gen   >> 8) & prop;
    flood |= gen = (gen   >> 8) & prop;
    flood |= gen = (gen   >> 8) & prop;
    flood |=       (gen   >> 8) & prop;
    return flood;
}

__host__ __device__
bitboard  EFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    prop &= notE;
    flood |= gen = (gen   << 1) & prop;
    flood |= gen = (gen   << 1) & prop;
    flood |= gen = (gen   << 1) & prop;
    flood |= gen = (gen   << 1) & prop;
    flood |= gen = (gen   << 1) & prop;
    flood |=       (gen   << 1) & prop;
    return                flood & notE;
}

__host__ __device__
bitboard NEFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    prop &= notE;
    flood |= gen = (gen   >> 7) & prop;
    flood |= gen = (gen   >> 7) & prop;
    flood |= gen = (gen   >> 7) & prop;
    flood |= gen = (gen   >> 7) & prop;
    flood |= gen = (gen   >> 7) & prop;
    flood |=       (gen   >> 7) & prop;
    return                flood & notE;
}

__host__ __device__
bitboard SEFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    prop &= notE;
    flood |= gen = (gen   << 9) & prop;
    flood |= gen = (gen   << 9) & prop;
    flood |= gen = (gen   << 9) & prop;
    flood |= gen = (gen   << 9) & prop;
    flood |= gen = (gen   << 9) & prop;
    flood |=       (gen   << 9) & prop;
    return                flood & notE;
}

__host__ __device__
bitboard  WFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    prop &= notW;
    flood |= gen = (gen   >> 1) & prop;
    flood |= gen = (gen   >> 1) & prop;
    flood |= gen = (gen   >> 1) & prop;
    flood |= gen = (gen   >> 1) & prop;
    flood |= gen = (gen   >> 1) & prop;
    flood |=       (gen   >> 1) & prop;
    return                flood & notW;
}

__host__ __device__
bitboard SWFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    prop &= notW;
    flood |= gen = (gen   << 7) & prop;
    flood |= gen = (gen   << 7) & prop;
    flood |= gen = (gen   << 7) & prop;
    flood |= gen = (gen   << 7) & prop;
    flood |= gen = (gen   << 7) & prop;
    flood |=       (gen   << 7) & prop;
    return                flood & notW;
}

__host__ __device__
bitboard NWFill(bitboard gen, bitboard prop) {
    bitboard flood = 0ULL;
    prop &= notW;
    flood |= gen = (gen   >> 9) & prop;
    flood |= gen = (gen   >> 9) & prop;
    flood |= gen = (gen   >> 9) & prop;
    flood |= gen = (gen   >> 9) & prop;
    flood |= gen = (gen   >> 9) & prop;
    flood |=       (gen   >> 9) & prop;
    return                flood & notW;
}

// Shift algorithms
__host__ __device__ bitboard  SShift (bitboard b) {return  b << 8;}
__host__ __device__ bitboard  NShift (bitboard b) {return  b >> 8;}
__host__ __device__ bitboard  EShift (bitboard b) {return (b << 1) & notE;}
__host__ __device__ bitboard SEShift (bitboard b) {return (b << 9) & notE;}
__host__ __device__ bitboard NEShift (bitboard b) {return (b >> 7) & notE;}
__host__ __device__ bitboard  WShift (bitboard b) {return (b >> 1) & notW;}
__host__ __device__ bitboard SWShift (bitboard b) {return (b << 7) & notW;}
__host__ __device__ bitboard NWShift (bitboard b) {return (b >> 9) & notW;}
