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


    Move *movesArr = new Move[60];
    int numMoves = b.getMovesAsArray(movesArr, side);
    moves = std::vector<Move>(movesArr, movesArr + numMoves);
    if (numMoves > 60) fprintf(stderr, "moves: %d\n", numMoves);
    children.resize(numMoves);

    if (numMoves == 0) {
        if (!board.isDone()) {
            fullyExpanded = false;
            moves.push_back(Move(-1, -1));
            children.resize(1);
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
    if (! (m.x == 1 && m.y == 1)) {
        b.doMove(moves[i], side);
    }
    children[i] = new Node(b, this, OTHER(side));
    return children[i];
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
        // fprintf(stderr, "Expanding child: (%d, %d)\n", moves[i].x, moves[i].y);
        return addChild(unvisited_children[i]);
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
            // fprintf(stderr, "Exploit: %f\t Explore: %f\n", exploit, explore);
            // float score = (float)n->winDiff / n->numSims
            //     + CP * sqrt(log((float)numSims) / (float)n->numSims);
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
    fprintf(stderr, "Played (%d, %d): %f, %d\n",
        bestScoreMove.x, bestScoreMove.y, bestScore, bestMoveFreq);
    return true;
}