#ifndef __COMMON_HPP__
#define __COMMON_HPP__

enum Side { 
    WHITE, BLACK
};

#define OTHER(x) (x == WHITE? BLACK:WHITE)

class Move {
  public:
    int x, y;
    Move() {
        this->x = -1;
        this->y = -1;
    }
    Move(int x, int y) {
        this->x = x;
        this->y = y;        
    }
    Move(const Move &m) {
        this->x = m.x;
        this->y = m.y;        
    }
    ~Move() {}
};

#endif
