#include "gametree.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <iostream>
#include <locale>

Node::Node(Board b, Node *parent, Side side) :
    board(b), parent(parent), side(side),
    winDiff(0), numSims(0), numDescendants(1), miniMaxScore(0), state(0) {

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
            state |= SCORE_FINAL;
            int pieceDiff = board.countPieces(side) - board.countPieces(OTHER(side));
            if (pieceDiff > 0) {
                state |= PROVEN_WIN;
                miniMaxScore = pieceDiff;
            }
            else if (pieceDiff < 0) {
                state |= PROVEN_LOSS;
                miniMaxScore = pieceDiff;
            }
            else {
                miniMaxScore = 0;
            }
        }
    }
    heuristicScore = board.getHeuristic(this->side);
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

Node *Node::searchScore(bool expand, bool useMinimax, bool maximizeScore) {
    // If this is a solved node, then just simulate from this node's parent
    // (if possible)
    if ((state & SCORE_FINAL) || (!maximizeScore && (state & SOLVED))) {
        if (this->parent) {
            return this->parent;
        }
        else {
            return this;
        }
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
        if (n && !((n->state & SCORE_FINAL) || (!maximizeScore && (n->state & SOLVED)))) {
            assert(n->numSims > 0);
            assert(n->parent == this);
            // Convert exploit to [-numSims, numSims] -> [0, 1]
            // and negate because it's the opponent's score
            float exploit = useMinimax?
                            (0.5 - n->miniMaxScore/2) :
                            ((float) (n->numSims - n->winDiff) / (2 * n->numSims));
            // Clamp exploit to be in [0, 1] (takes care of proven wins / losses, which
            // report piece differentials)
            if (exploit < 0) exploit = 0;
            if (exploit > 1) exploit = 1;
            float explore = sqrt(2 * log(adjustedNumSims())
                / n->adjustedNumSims());
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
    return bestChild->searchScore(expand, useMinimax, maximizeScore);
}

std::vector<Node *> Node::searchScoreBlock(bool expand, bool useMinimax, bool maximizeScore) {
    // If this is a terminal, then just simulate from this node's siblings
    if ((state & SCORE_FINAL) || (!maximizeScore && (state & SOLVED))) {
        if (this->parent) {
            return this->parent->children;
        }
        else {
            return {this};
        }
    }

    if (!(state & FULLY_EXPANDED) && expand) {
        return addChildren();
    }

    // Otherwise, choose child based on UCT score
    float bestScore = -1;
    Node *bestChild = nullptr;
    for (Node *n : children) {
        if (n && !((n->state & SCORE_FINAL) || (!maximizeScore && (n->state & SOLVED)))) {
            assert(n->numSims > 0);
            assert(n->parent == this);
            // Convert exploit to [-numSims, numSims] -> [0, 1]
            // and negate because it's the opponent's score
            float exploit = useMinimax?
                            0.5 * (1. - n->miniMaxScore) :
                            ((float) (n->numSims - n->winDiff) / (2 * n->numSims));
            // Clamp exploit to be in [0, 1] (takes care of proven wins / losses, which
            // report piece differentials)
            if (exploit < 0) exploit = 0;
            if (exploit > 1) exploit = 1;
            float explore = sqrt(2 *
                log(adjustedNumSims()) /
                (n->adjustedNumSims()));
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
    return bestChild->searchScoreBlock(expand, useMinimax, maximizeScore);
}


void Node::incrementNumDescendants(int numToAdd) {
    numDescendants += numToAdd;
    if (parent) parent->incrementNumDescendants(numToAdd);
}

uint Node::adjustedNumSims() {
    return HEURISTIC_PRIOR * numDescendants + numSims;
}

void Node::updateSim(int numSims, int winDiff, bool updateMinimaxScore) {
    this->winDiff += winDiff;
    this->numSims += numSims;
    assert(numDescendants != 0);

    assert(this->numSims > 0);

    if (this->state & SCORE_FINAL) {
        // Don't update anything
    }
    // If this node doesn't have any children, then set miniMaxScore to
    // the win rate (with heuristic prior)
    else if (numDescendants == 1) {
        this->miniMaxScore =
            (this->winDiff + this->heuristicScore * HEURISTIC_PRIOR)
                                / (this->numSims + HEURISTIC_PRIOR);
    }
    else if (updateMinimaxScore) {
        assert( children.size() > 0);
        // If ANY child is a proven loss, then this is a proven win
        // If ALL children are proven wins, then this is a proven loss.
        // If ALL children are solved (or this is a proven win),
        //  then this is also solved.
        // If ALL children have finalized scores, then this also does.
        this->state |= SOLVED;
        this->state |= PROVEN_LOSS;
        this->state &= ~PROVEN_WIN;
        this->state |= SCORE_FINAL;
        float max_score = -FLT_MAX;
        float n_score;
        for (Node *n : children) {
            if (n) {
                n_score = -n->miniMaxScore;
                if (!(n->state & SOLVED)) {
                    n_score *= (1. - sqrt(1.f / n->adjustedNumSims()));
                }
                if (n_score > max_score) {
                    max_score = n_score;
                    this->miniMaxScore = -n->miniMaxScore;
                }
                if (n->state & PROVEN_LOSS) {
                    this->state |= PROVEN_WIN;
                }
                if (!(n->state & PROVEN_WIN)) {
                    this->state &= ~PROVEN_LOSS;
                }
                if (!(n->state & SOLVED)) {
                    this->state &= ~SOLVED;
                }
                if (!(n->state & SCORE_FINAL)) {
                    this->state &= ~SCORE_FINAL;
                }
            }
            else {
                this->state &= ~PROVEN_LOSS;
                this->state &= ~SOLVED;
            }
        }
        if (this->state & PROVEN_WIN) {
            this->state |= SOLVED;
        }
        if (abs(this->miniMaxScore) > 64) {
            fprintf(stderr, "OoB score: %f\n", this->miniMaxScore);
        }
    }
    // Node can't be both a PROVEN_WIN and PROVEN_LOSS
    assert( !((this->state & PROVEN_WIN) && (this->state & PROVEN_LOSS)) );
    // Negate number of wins
    if (parent) parent->updateSim(numSims, -winDiff, updateMinimaxScore);
}


bool Node::getBestMove(Move *m, bool useMinimax, bool forceResult) {
    float bestScore = -FLT_MAX;
    Move bestScoreMove(-1, -1);
    uint bestMoveFreq = 0;
    Move mostFrequentMove(-1, -1);
    Move moves[MAX_NUM_MOVES];
    int numMoves = board.getMovesAsArray(moves, side);
    float score;
    for (uint i = 0; i < numMoves; i++) {
        Node *n = children[i];
        if (n) {
            assert(n->side != side);
            if (useMinimax) {
                score = -n->miniMaxScore;
                if (!(n->state & SOLVED)) {
                    score *= (1. - sqrt(1.f / n->adjustedNumSims()));
                }
            }
            else {
                score = (float) -n->winDiff / n->numSims;
            }
            if (state & PROVEN_WIN) {
                // If any child is a proven loss for the other side, then choose that move.
                if (n->state & PROVEN_LOSS) {
                    if (!(state & PROVEN_WIN) || -n->miniMaxScore > bestScore) {
                        bestScoreMove = moves[i];
                        bestScore = -n->miniMaxScore;
                        mostFrequentMove = moves[i];
                        bestMoveFreq = n->numSims;
                    }
                }
            }
            else {
                // If the current node is a guaranteed tie (and not a guaranteed
                // win), then choose the child that is also a guaranteed tie.
                if ((this->state & SOLVED) &&
                    (n->state & SOLVED) && !(n->state & PROVEN_WIN)) {
                    bestScoreMove = moves[i];
                    mostFrequentMove = moves[i];
                    bestScore = score;
                    bestMoveFreq = n->numSims;
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
    }
    if (bestScoreMove != mostFrequentMove
        && !(state & SCORE_FINAL)
        && !forceResult) {
        return false;
    }

    *m = bestScoreMove;
    if (state & SCORE_FINAL) {
        std::cerr << "[Final score] ";
    }
    else if (state & SOLVED) {
        std::cerr << "[Proven] ";
    }
    fprintf(stderr, "Played (%u, %u): %f (%.1e / %.1e)\n",
        bestScoreMove.x, bestScoreMove.y,
        bestScore, (float)bestMoveFreq, (float)numSims);
    return true;
}
