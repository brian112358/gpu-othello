#include "gametree.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>

Node::Node(Board b, Node *parent, Side side) :
    board(b), parent(parent), side(side),
    winDiff(0), numSims(0), numDescendants(1), miniMaxScore(0), state(0) {

    if ((unsigned long long) this > 0xffffffff) {
        fprintf(stderr, "Weird node pointer: %p\n", this);
    }

    Move movesArr[MAX_NUM_MOVES];
    int numMoves = b.getMovesAsArray(movesArr, side);
    assert( 0 <= numMoves && numMoves < MAX_NUM_MOVES);
    children.resize(numMoves, nullptr);

    if (numMoves == 0) {
        if (!board.isDone()) {
            children.resize(1, nullptr);
        }
        else {
            state |= FULLY_EXPANDED;
            state |= SOLVED;
            int pieceDiff = board.countPieces(side) - board.countPieces(OTHER(side));
            if (pieceDiff > 0) {
                state |= PROVEN_WIN;
                miniMaxScore = 1;
            }
            else if (pieceDiff < 0) {
                state |= PROVEN_LOSS;
                miniMaxScore = -1;
            }
            else {
                miniMaxScore = 0;
            }
        }
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
    incrementNumDescendants(1);
    return children[i];
}

std::vector<Node *> Node::addChildren() {
    Board b;
    Move moves[MAX_NUM_MOVES];
    int numMoves = this->board.getMovesAsArray(moves, side);

    if (numMoves > 0) {
        for (uint i = 0; i < numMoves; i++) {
            b = this->board;
            b.doMove(moves[i], side);
            assert (children[i] == nullptr);
            children[i] = new Node(b, this, OTHER(side));
        }
    }
    else {
        assert (children[0] == nullptr);
        children[0] = new Node(this->board, this, OTHER(side));
    }
    incrementNumDescendants(numMoves > 0? numMoves:1);
    this->state |= FULLY_EXPANDED;
    return children;
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

Node *Node::searchScore(bool expand, bool useMinimax) {
    if (state & SOLVED) {
        // If this is a solved node, then just simulate from this node
        return this;
    }

    if (!(state & FULLY_EXPANDED) && expand) {
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
            state |= FULLY_EXPANDED;
        }

        assert(unvisited_children.size() > 0);

        int i = rand() % unvisited_children.size();
        return addChild(unvisited_children[i]);
    }

    // Otherwise, choose child based on UCT score
    float bestScore = -1;
    Node *bestChild = nullptr;
    for (Node *n : children) {
        // Don't choose children that have already been solved.
        if (n && !(n->state & SOLVED)) {
            assert(n->numSims > 0);
            assert(n->parent == this);
            // Convert exploit to [-numSims, numSims] -> [0, 1]
            // and negate because it's the opponent's score
            float exploit = useMinimax?
                            (0.5 - n->miniMaxScore/2) :
                            ((float) (n->numSims - n->winDiff) / (2 * n->numSims));
            assert (-1e-6 <= exploit && exploit <= 1 + 1e-6);
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
    return bestChild->searchScore(expand, useMinimax);
}

std::vector<Node *> Node::searchScoreBlock(bool expand, bool useMinimax) {
    if (state & SOLVED) {
        // If this is a terminal, then just simulate from this node
        return {this};
    }

    if (!(state & FULLY_EXPANDED) && expand) {
        return addChildren();
    }

    // Otherwise, choose child based on UCT score
    float bestScore = -1;
    Node *bestChild = nullptr;
    for (Node *n : children) {
        if (n && !(n->state & SOLVED)) {
            assert(n->numSims > 0);
            assert(n->parent == this);
            // Convert exploit to [-numSims, numSims] -> [0, 1]
            // and negate because it's the opponent's score
            float exploit = useMinimax?
                            0.5 * (1. - n->miniMaxScore) :
                            ((float) (n->numSims - n->winDiff) / (2 * n->numSims));
            // assert (-1e-6 <= exploit && exploit <= 1 + 1e-6);
            if (!(-0.1 <= exploit && exploit <= 1.1)) {
                fprintf(stderr, "Exploit not within bounds: %f\n", exploit);
            }
            float explore = sqrt(2 * log((float)numSims) / n->numSims);
            float score = exploit + CP * explore;
            if (!bestChild || score > bestScore) {
                bestScore = score;
                bestChild = n;
            }
        }
    }
    // If there are no eligible children, just return this node
    if (!bestChild) {
        return {this};
    }
    return bestChild->searchScoreBlock(expand, useMinimax);
}


void Node::incrementNumDescendants(int numToAdd) {
    numDescendants += numToAdd;
    if (parent) parent->incrementNumDescendants(numToAdd);
}

void Node::updateSim(int numSims, int winDiff) {
    this->winDiff += winDiff;
    this->numSims += numSims;
    assert(numDescendants != 0);

    if (this->state & SOLVED) { 
        if (this->state & PROVEN_WIN) {
            this->miniMaxScore = 1;
        }
        else if (this->state & PROVEN_LOSS) {
            this->miniMaxScore = -1;
        }
        else {
            this->miniMaxScore = 0;
        }
    }
    // If this node doesn't have any children, then set miniMaxScore to 
    // the win rate (possibly add heuristic as a prior here)
    else if (numDescendants == 1) {
        assert (! (this->state & PROVEN_WIN) && ! (this->state & PROVEN_LOSS));
        float h = board.getHeuristic(this->side);
        if (!(-1 - 1e-6 <= h && h <= 1 + 1e-6)) {
            fprintf(stderr, "h not within bounds: %f\n", h);
        }
        this->miniMaxScore = (this->winDiff + h * HEURISTIC_PRIOR)
                                / (this->numSims + HEURISTIC_PRIOR);
    }
    else {
        assert( children.size() > 0);
        this->miniMaxScore = -FLT_MAX;
        // If ANY child is a proven loss, then this is a proven win
        // If ANY children are proven wins, then this is a proven loss.
        this->state |= PROVEN_LOSS;
        this->state &= ~PROVEN_WIN;
        this->state &= ~SOLVED;
        for (Node *n : children) {
            if (n) {
                if (-n->miniMaxScore > this->miniMaxScore) {
                    this->miniMaxScore = -n->miniMaxScore;
                }
                if (n->state & PROVEN_LOSS) {
                    this->state |= PROVEN_WIN;
                }
                if (!(n->state & PROVEN_WIN)) {
                    this->state &= ~PROVEN_LOSS;
                }
            }
            else {
                this->state &= ~PROVEN_LOSS;
            }
        }
        assert(this->miniMaxScore > -1000);
    }
    if (this->state & PROVEN_WIN) {
        if (this->miniMaxScore != 1) {
            fprintf(stderr, "PROVEN_WIN: score = %f\n", this->miniMaxScore);
        }
    }
    if (this->state & PROVEN_LOSS) {
        if (this->miniMaxScore != -1) {
            fprintf(stderr, "PROVEN_LOSS: score = %f\n", this->miniMaxScore);
        }
    }
    // Node can't be both a PROVEN_WIN and PROVEN_LOSS
    assert( !(this->state & PROVEN_WIN) || !(this->state & PROVEN_LOSS) );
    // Negate number of wins 
    if (parent) parent->updateSim(numSims, -winDiff);
}


bool Node::getBestMove(Move *m, bool useMinimax, bool forceResult) {
    float bestScore = -FLT_MAX;
    Move bestScoreMove(-1, -1);
    int bestMoveFreq = 0;
    Move mostFrequentMove(-1, -1);
    Move moves[MAX_NUM_MOVES];
    int numMoves = board.getMovesAsArray(moves, side);
    for (uint i = 0; i < numMoves; i++) {
        Node *n = children[i];
        if (n) {
            assert(n->side != side);
            float score = useMinimax? (-n->miniMaxScore) : ((float) -n->winDiff / n->numSims);
            // If any child is a proven loss for the other side, then choose that move.
            if (n->state & PROVEN_LOSS) {
                bestScoreMove = moves[i];
                mostFrequentMove = moves[i];
                bestScore = score;
                bestMoveFreq = n->numSims;
                fprintf(stderr, "Proven win!\n");
                break;
            }
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
    if (bestScoreMove != mostFrequentMove && !forceResult) {
        fprintf(stderr, "Frequency-score mismatch, re-searching...\n");
        return false;
    }

    *m = bestScoreMove;
    fprintf(stderr, "Played (%d, %d): %f (%d/%d)\n",
        bestScoreMove.x, bestScoreMove.y, bestScore, bestMoveFreq, numSims);
    return true;
}