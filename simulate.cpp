#include "simulate.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "board.hpp"
#include "common.hpp"

#include <cuda_runtime.h>
#include <curand.h>

// Experimentally verified to stay within 768MB memory limit
#define MAX_NUM_NODES 600000

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    cudaEventCreate(&start);         \
    cudaEventCreate(&stop);          \
    cudaEventRecord(start);          \
}

#define STOP_RECORD_TIMER(name) {                           \
    cudaEventRecord(stop);                       \
    cudaEventSynchronize(stop);                  \
    cudaEventElapsedTime(&name, start, stop);    \
    cudaEventDestroy(start);                     \
    cudaEventDestroy(stop);                      \
}

int simulateNode(Node *n, Move *moves) {
    Board b = n->board;
    Side side = n->side;
    int numPasses = 0;
    while (numPasses < 2) {
        int numMoves = b.getMovesAsArray(moves, side);
        if (numMoves) {
            b.doMove(moves[rand() % numMoves], side);
            numPasses = 0;
        }
        else {
            numPasses++;
        }
        side = OTHER(side);
    }
    int pieceDiff = b.countPieces(n->side) - b.countPieces(OTHER(n->side));
    return (pieceDiff > 0)? 1:( (pieceDiff == 0)? 0:-1);
}

int expandGameTree(Node *root, int ms) {
    clock_t startTime = clock();

    Move moves[MAX_NUM_MOVES];
    Node *n;
    int numSims = 0;
    int outcome;

    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000) {
        n = root->searchScore(root->numDescendants < MAX_NUM_NODES);
        outcome = simulateNode(n, moves);
        n->updateSim(1, outcome);
        numSims++;
    }
    return numSims;
}


__global__
void cudaSimulateGameKernel(Board board, Side side, int numSims,
    float *rands, int *winDiff) {
    // Shared memory for partial sums
    extern __shared__ int partial_winDiff[];

    const uint tid = threadIdx.x;
    const uint idx = blockIdx.x * blockDim.x + tid;
    const uint total_threads = blockDim.x * gridDim.x;

    int piecesDiff;

    int numMoves;
    uint moveNum;
    uint moveIdx;

    Board b;
    Move moves[MAX_NUM_MOVES];
    Side simSide;
    int numPasses;

    // Initialize winDiffs to 0
    partial_winDiff[tid] = 0;

    __syncthreads();

    for (uint i = idx; i < numSims; i += total_threads) {
        b = board;
        simSide = side;
        moveNum = 0;
        numPasses = 0;
        while (numPasses < 2) {
            numMoves = b.getMovesAsArray(moves, simSide);
            if (numMoves) {
                moveIdx = (uint) (rands[numSims * moveNum + i] * numMoves);
                b.doMove(moves[moveIdx], simSide);
                numPasses = 0;
                moveNum++;
            }
            else {
                numPasses++;
            }
            simSide = OTHER(simSide);
        }
        piecesDiff = b.countPieces(side) - b.countPieces(OTHER(side));
        if (piecesDiff > 0) {
            partial_winDiff[tid] += 1;
        }
        else if (piecesDiff < 0) {
            partial_winDiff[tid] -= 1;
        }
    }

    // Reduction step
    __syncthreads();
    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            partial_winDiff[tid] += partial_winDiff[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(winDiff, partial_winDiff[0]);
}

#define CPU_SIMS_PER_ITER 2
#define GPU_SIMS_PER_ITER 1024
#define NBLOCKS 8
#define NTHREADS 128

void startNodeSimulationGpu(Node *n, int *winDiff, int *d_winDiff,
        float *d_rands, curandGenerator_t &gen, int turnsLeft,
        uint nBlocks, uint nThreads, cudaStream_t stream) {

    // Generate random numbers for the simulations
    curandGenerateUniform(gen, d_rands, turnsLeft * GPU_SIMS_PER_ITER);

    // Initialize the win difference to 0
    cudaMemset(d_winDiff, 0, 1 * sizeof(int));

    // Run simulation kernel
    cudaSimulateGameKernel<<<nBlocks, nThreads, nThreads*sizeof(int), stream>>>(
            n->board, n->side, GPU_SIMS_PER_ITER, d_rands, d_winDiff);

    // Copy result back to CPU
    cudaMemcpyAsync(winDiff, d_winDiff,
        1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
}

int expandGameTreeGpu(Node *root, int ms) {
    clock_t startTime = clock();

    const int turnsLeft = root->board.countEmpty();

    // Random numbers used to calculate random game states taken
    float *d_rands;

    int *winDiff;
    int *d_winDiff;

    cudaError err;

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // // Allocate memory on GPU
    try {
        gpuErrchk(cudaMalloc(&d_rands,
            turnsLeft * GPU_SIMS_PER_ITER * sizeof(float)));
    }
    catch(std::bad_alloc&) {
        fprintf(stderr, "bad_alloc exception handled in cudaMalloc.\n");
    }
    gpuErrchk(cudaMalloc(&d_winDiff, 1 * sizeof(int)));
    gpuErrchk(cudaMallocHost(&winDiff, 1 * sizeof(int)));

    Node *nGpu = nullptr;
    
    // CPU simulation variables
    Node *nCpu;
    int cpuOutcome;
    Move moves[MAX_NUM_MOVES];
    int numIters = 0;
    
    // Main loop
    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000) {
        if (cudaStreamQuery(stream) == cudaSuccess) {
            // Update using the last result
            if (nGpu) {
                nGpu->updateSim(GPU_SIMS_PER_ITER-1, *winDiff);
            }
            nGpu = root->searchScore(root->numDescendants < MAX_NUM_NODES);
            // Put dummy result at node
            nGpu->updateSim(1, 0);
            
            startNodeSimulationGpu(nGpu, winDiff, d_winDiff,
                d_rands, gen, turnsLeft, NBLOCKS, NTHREADS, stream);
            // cudaStreamSynchronize(stream);

            err = cudaGetLastError();
            if ( cudaSuccess != err ) {
                fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                         __FILE__, __LINE__, cudaGetErrorString( err ) );
                exit( -1 );
            }
        }
        else {
            nCpu = root->searchScore(root->numDescendants < MAX_NUM_NODES);
            cpuOutcome = 0;
            for (uint i = 0; i < CPU_SIMS_PER_ITER; i++) {
                cpuOutcome += simulateNode(nCpu, moves);
            }
            nCpu->updateSim(CPU_SIMS_PER_ITER, cpuOutcome);
        }

        numIters++;
    }

    // Finish the last GPU call
    cudaStreamSynchronize(stream);
    if (nGpu) nGpu->updateSim(GPU_SIMS_PER_ITER-1, *winDiff);

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // Free memory
    curandDestroyGenerator(gen);
    cudaStreamDestroy(stream);
    try {
        gpuErrchk(cudaFree(d_rands));
    }
    catch(std::bad_alloc&) {
        fprintf(stderr, "bad_alloc exception handled in cudaFree.\n");
    }
    gpuErrchk(cudaFree(d_winDiff));
    gpuErrchk(cudaFreeHost(winDiff));

    return numIters;
}
