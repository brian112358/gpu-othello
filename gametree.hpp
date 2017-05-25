#ifndef __GAMETREE_HPP__
#define __GAMETREE_HPP__

#include <vector>
#include "board.hpp"
#include "common.hpp"

// Cp is a constant used for calculating a node's UCT score, choose 1/sqrt(2)
#define CP 0.707

class Node {
  public:
    Side side;
    Board board;
    Node *parent;
    bool terminal;
    bool fullyExpanded;
    // std::vector<Move> moves;
    std::vector<Node*> children;

    int winDiff;
    uint numSims;

    uint numDescendants;

    Node(Board b, Node *parent, Side side);
    ~Node();


    // Search all descendants of this node for the best node to expand
    // based on UCT score.
    Node *searchScore(bool expand);

    Node *searchBoard(Board b, Side s, int depth);

    void updateSim(int numSims, int winDiff);


    // Given simulations so far, return move with most number of simulations.
    bool getBestMove(Move *m);

  private:
    // Increment the number of descendants for all ancestors of this node. 
    void incrementNumDescendants();

    // Create Node for the given child index (same index as move).
    Node *addChild(int i);
};

#endif