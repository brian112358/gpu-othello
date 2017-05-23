#include "gametree.hpp"

#include <cmath>
#include <cstdlib>

Node::Node(Board b, Node *parent, Side side) {
    this->board = b;
    this->parent = parent;
    this->side = side;
    numWins = 0;
    numSims = 0;


    Move *movesArr = new Move[60];
    int numMoves = b.getMovesAsArray(movesArr, side);
    moves = std::vector<Move>(movesArr, movesArr + numMoves);
    children.resize(numMoves);

    if (numMoves == 0) {
        fullyExpanded = true;
        if (!board.isDone()) {
            moves.push_back(Move(-1, -1));
            children.push_back(new Node(board, this, OTHER(side)));
        }
        else {
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
    b.doMove(moves[i], side);
    children[i] = new Node(b, this, OTHER(side));
    return children[i];
}

Node *Node::searchScore() {
    if (!fullyExpanded) {
        // Then randomly expand one of the children
        std::vector<int> unvisited_children;
        for (uint i = 0; i < moves.size(); i++) {
            if (!children[i]) unvisited_children.push_back(i);
        }
        return addChild(unvisited_children[rand() % unvisited_children.size()]);
    }
    else if (!terminal) {
        // Choose child based on UCT score
        float bestScore = -1;
        Node *bestChild = nullptr;
        for (Node *n : children) {
            float score = n->numWins / n->numSims + CP * sqrt(log(numSims) / n->numSims);
            if (!bestChild || score > bestScore) {
                bestScore = score;
                bestChild = n;
            }
        }
        return bestChild->searchScore();
    }
    else {
        return nullptr;
    }
}

void Node::updateSim(int numSims, int numWins) {
    this->numWins += numWins;
    this->numSims += numSims;
    if (parent) parent->updateSim(numSims, numWins);
}

// Currently will fail if there are no explored moves
Move Node::getBestMove() {
    int bestMoveCount = 0;
    Move bestMove(-1, -1);
    for (uint i = 0; i < moves.size(); i++) {
        if (children[i] && children[i]->numSims > bestMoveCount) {
            bestMove = moves[i];
        }
    }
    return bestMove;
}
