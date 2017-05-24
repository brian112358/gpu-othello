#include "gametree.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>

Node::Node(Board b, Node *parent, Side side) {
    this->board = b;
    this->parent = parent;
    this->side = side;
    winDiff = 0;
    numSims = 0;

    if ((unsigned long long) this > 0xffffffff) {
        fprintf(stderr, "Weird node pointer: %p\n", this);
    }

    Move *movesArr = new Move[32];
    int numMoves = b.getMovesAsArray(movesArr, side);
    assert( 0 <= numMoves && numMoves < 32);
    moves = std::vector<Move>(movesArr, movesArr + numMoves);
    children.resize(numMoves, nullptr);

    if (numMoves == 0) {
        if (!board.isDone()) {
            fullyExpanded = false;
            terminal = false;
            moves.push_back(Move(-1, -1));
            children.resize(1, nullptr);
        }
        else {
            fullyExpanded = true;
            terminal = true;
        }
    }
    else {
        terminal = false;
        fullyExpanded = false;
    }
    delete[] movesArr;
}

Node::~Node() {
    for(Node *n : children) {
        if (n) delete n;
    }
}

Node *Node::addChild(int i) {
    Board b = this->board;
    Move m = moves[i];
    // If move is PASS do nothing
    if (m != Move(-1, -1)) {
        b.doMove(moves[i], side);
    }
    assert (children[i] == nullptr);
    // try {
        children[i] = new Node(b, this, OTHER(side));
    // }
    // catch(std::bad_alloc&) {
    //     fprintf(stderr, "bad_alloc exception handled in addChild new.\n");
    // }
    return children[i];
}

// Searches two-depth down 
Node *Node::searchBoard(Board b, Side s, int depth) {
    if (depth < 0) {
        return nullptr;
    }
    if (this->board == b && this->side == s) {
        return this;
    }
    for (Node *n : children) {
        if (n) {
            Node *result = n->searchBoard(b, s, depth-1);
            if (result) return result;
        }
    }
    return nullptr;
}

Node *Node::searchScore() {
    if (!fullyExpanded) {
        // Then randomly expand one of the children
        std::vector<int> unvisited_children;
        if (moves.size() == 1 && children.size() == 0) {
            fullyExpanded = true;
            unvisited_children.push_back(0);
        }
        else {
            for (uint i = 0; i < moves.size(); i++) {
                if (!children[i]) {
                    unvisited_children.push_back(i);
                }
                else {
                    assert(children[i]->numSims > 0);
                }
            }
        }
        if (unvisited_children.size() == 1) {
            fullyExpanded = true;
        }

        int i = rand() % unvisited_children.size();
        Node *n;
        // try {
            n = addChild(unvisited_children[i]);
        // }
        // catch(std::bad_alloc&) {
        //     fprintf(stderr, "bad_alloc exception handled in addChild.\n");
        // }
        return n;
    }
    else if (!terminal) {
        // Choose child based on UCT score
        float bestScore = -1;
        Node *bestChild = nullptr;
        for (Node *n : children) {
            assert( n );
            assert( n->numSims > 0 );
            float exploit = (float) -n->winDiff / n->numSims;
            float explore = (float) sqrt(log((float)numSims) / n->numSims);
            float score = exploit + CP * explore;
            if (!bestChild || score > bestScore) {
                bestScore = score;
                bestChild = n;
            }
        }
        if (!bestChild) {
            fprintf(stderr, "No children\n");
            return this;
        }
        return bestChild->searchScore();
    }
    else {
        // If this is a terminal and fully expanded node, then just simulate
        // from this node
        return this;
    }
}

void Node::updateSim(int numSims, int winDiff) {
    this->winDiff += winDiff;
    this->numSims += numSims;
    // Negate number of wins 
    if (parent) {
        parent->updateSim(numSims, -winDiff);
    }
}

bool Node::getBestMove(Move *m) {
    float bestScore = -FLT_MAX;
    Move bestScoreMove(-1, -1);
    int bestMoveFreq = 0;
    Move mostFrequentMove(-1, -1);
    for (uint i = 0; i < moves.size(); i++) {
        Node *n = children[i];
        if (n) {
            float score = (float) -n->winDiff / n->numSims;
            if (score > bestScore) {
                bestScoreMove = moves[i];
                bestScore = score;
            }
            if (n->numSims > bestMoveFreq) {
                mostFrequentMove = moves[i];
                bestMoveFreq = n->numSims;
            }
        }
    }
    if (bestScoreMove != mostFrequentMove) {
        fprintf(stderr, "Frequency-score mismatch, re-searching...\n");
        return false;
    }

    *m = bestScoreMove;
    fprintf(stderr, "Played (%d, %d): %f (%d/%d)\n",
        bestScoreMove.x, bestScoreMove.y, bestScore, bestMoveFreq, numSims);
    return true;
}