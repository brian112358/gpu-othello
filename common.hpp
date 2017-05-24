#ifndef __COMMON_HPP__
#define __COMMON_HPP__

#include <cuda.h>
#include <cuda_runtime.h>

enum Side { 
    WHITE, BLACK
};

#define OTHER(x) (x == WHITE? BLACK:WHITE)

class Move {
  public:
    int x, y;
    __host__ __device__ Move() {
        this->x = -1;
        this->y = -1;
    }
    __host__ __device__ Move(int x, int y) {
        this->x = x;
        this->y = y;        
    }
    __host__ __device__ Move(const Move &m) {
        this->x = m.x;
        this->y = m.y;
    }
    __host__ __device__ bool operator==(const Move &other) const {
        return this->x == other.x && this->y == other.y;
    }
    __host__ __device__ bool operator!=(const Move &other) const {
        return !(*this == other);
    }
    __host__ __device__ ~Move() {}
};

#endif
