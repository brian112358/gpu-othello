#ifndef __GAMETREE_HPP__
#define __GAMETREE_HPP__

#include <vector>
#include "board.hpp"
#include "common.hpp"

// Cp is a constant used for calculating a node's UCT score, choose 1/sqrt(2)
#define CP 0.707

#define HEURISTIC_PRIOR 10

// Store binary states in a single byte; following are the bitmasks
#define SOLVED 0x01
#define PROVEN_WIN 0x02
#define PROVEN_LOSS 0x04
#define FULLY_EXPANDED 0x08

class Node {
  public:
    Side side;
    Board board;
    Node *parent;

    unsigned char state;

    std::vector<Node*> children;

    int winDiff;
    uint numSims;

    uint numDescendants;
    float miniMaxScore;

    Node(Board b, Node *parent, Side side);
    ~Node();


    // Search all descendants of this node for the best node to expand
    // based on UCT score.
    Node *searchScore(bool expand, bool useMinimax);

    std::vector<Node *> searchScoreBlock(bool expand, bool useMinimax);

    Node *searchBoard(Board b, Side s, int depth);

    void updateSim(int numSims, int winDiff);


    // Given simulations so far, return move with most number of simulations.
    bool getBestMove(Move *m, bool useMinimax, bool forceResult);

  private:
    // Increment the number of descendants for all ancestors of this node.
    void incrementNumDescendants(int numToAdd);
    void incrementNumDescendants();

    // Create Node for the given child index (same index as move).
    Node *addChild(int i);

    std::vector<Node *> addChildren();
};

#endif