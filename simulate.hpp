#ifndef __SIMULATE_HPP__
#define __SIMULATE_HPP__

#include "gametree.hpp"

int expandGameTree(Node *root, bool useMinimax, int ms);

int expandGameTreeGpu(Node *root, bool useMinimax, int ms);

int expandGameTreeGpuBlock(Node *root, bool useMinimax, int ms);

#endif