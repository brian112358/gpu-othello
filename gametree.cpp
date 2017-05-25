#include "gametree.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>

Node::Node(Board b, Node *parent, Side side) :
    board(b), parent(parent), side(side),
    winDiff(0), numSims(0), numDescendants(0) {

    if ((unsigned long long) this > 0xffffffff) {
        fprintf(stderr, "Weird node pointer: %p\n", this);
    }

    Move movesArr[MAX_NUM_MOVES];
    int numMoves = b.getMovesAsArray(movesArr, side);
    assert( 0 <= numMoves && numMoves < MAX_NUM_MOVES);
    // moves = std::vector<Move>(movesArr, movesArr + numMoves);
    children.resize(numMoves, nullptr);

    if (numMoves == 0) {
        if (!board.isDone()) {
            fullyExpanded = false;
            terminal = false;
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
}

Node::~Node() {
    for(Node *n : children) {
        if (n) delete n;
    }
}

Node *Node::addChild(int i) {
    Board b = this->board;
    Move moves[MAX_NUM_MOVES];
    int numMoves = b.getMovesAsArray(moves, side);
    assert(0 <= i && i <= numMoves);
    // If move is PASS do nothing
    if (numMoves > 0) {
        b.doMove(moves[i], side);
    }
    assert (children[i] == nullptr);
    children[i] = new Node(b, this, OTHER(side));
    children[i]->incrementNumDescendants();
    return children[i];
}

// Searches depth down 
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

Node *Node::searchScore(bool expand) {
    if (terminal) {
        // If this is a terminal and fully expanded node, then just simulate
        // from this node
        return this;
    }

    if (!fullyExpanded && expand) {
        // If node is not fully expanded,
        // then randomly expand one of the children
        std::vector<int> unvisited_children;
        for (uint i = 0; i < children.size(); i++) {
            if (!children[i]) {
                unvisited_children.push_back(i);
            }
            else {
                assert(children[i]->numSims > 0);
            }
        }
        if (unvisited_children.size() == 1) {
            fullyExpanded = true;
        }

        assert(unvisited_children.size() > 0);

        int i = rand() % unvisited_children.size();
        return addChild(unvisited_children[i]);
    }

    // Otherwise, choose child based on UCT score
    float bestScore = -1;
    Node *bestChild = nullptr;
    for (Node *n : children) {
        if (n) {
            assert(n->numSims > 0);
            assert(n->parent == this);
            // Convert exploit to [-numSims, numSims] -> [0, 1]
            // and negate because it's the opponent's winDiff
            float exploit = (float) (n->numSims - n->winDiff) / (2 * n->numSims);
            // assert (0 <= exploit && exploit <= 1);
            float explore = sqrt(2 * log((float)numSims) / n->numSims);
            float score = exploit + CP * explore;
            if (!bestChild || score > bestScore) {
                bestScore = score;
                bestChild = n;
            }
        }
    }
    // If there are no eligible children, just return this
    if (!bestChild) {
        return this;
    }
    return bestChild->searchScore(expand);
}


void Node::incrementNumDescendants() {
    numDescendants++;
    if (parent) parent->incrementNumDescendants();
}

void Node::updateSim(int numSims, int winDiff) {
    this->winDiff += winDiff;
    this->numSims += numSims;
    // Negate number of wins 
    if (parent) parent->updateSim(numSims, -winDiff);
}


bool Node::getBestMove(Move *m) {
    float bestScore = -FLT_MAX;
    Move bestScoreMove(-1, -1);
    int bestMoveFreq = 0;
    Move mostFrequentMove(-1, -1);
    Move moves[MAX_NUM_MOVES];
    int numMoves = board.getMovesAsArray(moves, side);
    for (uint i = 0; i < numMoves; i++) {
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