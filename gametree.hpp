#ifndef __GAMETREE_HPP__
#define __GAMETREE_HPP__

#include <vector>
#include "board.hpp"
#include "common.hpp"

// Cp is a constant used for calculating a node's UCT score
#define CP 1.4

class Node {
  public:
    Side side;
    Board board;

    Node(Board b, Node *parent, Side side);
    ~Node();

    // Create Node for the given Child index (same index as move).
    Node *addChild(int i);

    // Search all descendants of this node for the best node to expand
    // based on UCT score. The max score will be returned, and the
    // node itself will be returned using the output variable expandNode.
    Node *searchScore();

    void updateSim(int numSims, int winDiff);

    // Given simulations so far, return move with most number of simulations.
    bool getBestMove(Move *m);

  private:
    bool terminal;
    bool fullyExpanded;
    Node *parent;
    std::vector<Move> moves;
    std::vector<Node*> children;
    int winDiff;
    int numSims;
};

#endif