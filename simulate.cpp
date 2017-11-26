#include "simulate.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "board.hpp"
#include "common.hpp"

#include <cuda_runtime.h>
#include <curand.h>

// #define DEBUG

// Experimentally verified to stay within 768MB memory limit
// #define MAX_NUM_NODES 500000

// Modified memory limit to 16GB
#define MAX_NUM_NODES 64000000

#define CPU_SIMS_PER_ITER 1

#define GPU_KERNEL_LAUNCH_SPACING 1
// TODO: running multiple streams seems to be blocking
#define GPU_NUM_KERNELS 1
#define NBLOCKS 16
#define NTHREADS 256

#define CPU_BLOCK_SIMS 1
#define GPU_BLOCK_SIMS 1024

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

int expandGameTree(Node *root, bool useMinimax, int ms) {
    clock_t startTime = clock();

    Move moves[MAX_NUM_MOVES];
    Node *n;
    int numSims = 0;
    int outcome;

    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000) {
        n = root->searchScore(root->numDescendants < MAX_NUM_NODES, useMinimax, root->state & SOLVED);
        outcome = simulateNode(n, moves);
        n->updateSim(1, outcome, true);
        numSims++;
    }
    return numSims;
}


__global__
void cudaSimulateGameKernel(Board board, Side side, int simsPerBlock,
    float *rands, int *winDiff) {
    // Shared memory for partial sums
    extern __shared__ int partial_winDiff[];

    uint tid = threadIdx.x;

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

    for (uint tid = threadIdx.x; tid < simsPerBlock; tid += blockDim.x) {
        b = board;
        simSide = side;
        moveNum = 0;
        numPasses = 0;
        while (numPasses < 2) {
            numMoves = b.getMovesAsArray(moves, simSide);
            // numMoves = b.getHeuristicMovesAsArray(moves, simSide);
            if (numMoves) {
                // rands[] is indexed by (moveNum, blockNum, tid)
                moveIdx = (uint) (rands[(moveNum * gridDim.x + blockIdx.x) * simsPerBlock + tid] * numMoves);
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
        if (threadIdx.x < s)
            partial_winDiff[threadIdx.x] += partial_winDiff[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(winDiff, partial_winDiff[0]);
}

void startNodeSimulationGpu(Node *n, int *winDiff, int *d_winDiff,
        float *d_rands, curandGenerator_t &gen, int turnsLeft,
        uint nBlocks, uint nThreads, cudaStream_t &stream) {

    // Generate random numbers for the simulations
    curandGenerateUniform(gen, d_rands, turnsLeft * nBlocks * nThreads);

    // Initialize the win difference to 0
    cudaMemsetAsync(d_winDiff, 0, 1 * sizeof(int), stream);

    // Run simulation kernel
    cudaSimulateGameKernel<<<nBlocks, nThreads, nThreads*sizeof(int), stream>>>(
            n->board, n->side, nThreads, d_rands, d_winDiff);

    // Copy result back to CPU
    cudaMemcpyAsync(winDiff, d_winDiff,
        1 * sizeof(int), cudaMemcpyDeviceToHost, stream);
}

int expandGameTreeGpu(Node *root, bool useMinimax, int ms) {
    clock_t startTime = clock();

    const int turnsLeft = root->board.countEmpty();

    #ifdef DEBUG
    cudaError err;
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    cudaStream_t streams[GPU_NUM_KERNELS];

    Node *nGpu[GPU_NUM_KERNELS];
    int *winDiff[GPU_NUM_KERNELS];
    int *d_winDiff[GPU_NUM_KERNELS];

    // Random numbers used to calculate random game states taken
    float *d_rands[GPU_NUM_KERNELS];
    curandGenerator_t gen[GPU_NUM_KERNELS];

    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        cudaStreamCreate(&streams[i]);

        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetStream(gen[i], streams[i]);

        // // Allocate memory on GPU
        gpuErrchk(cudaMalloc(&d_rands[i],
            turnsLeft * NBLOCKS * NTHREADS * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_winDiff[i], 1 * sizeof(int)));
        gpuErrchk(cudaMallocHost(&winDiff[i], 1 * sizeof(int)));

        nGpu[i] = nullptr;
    }

    // CPU simulation variables
    Node *nCpu;
    int cpuOutcome;
    Move moves[MAX_NUM_MOVES];
    int numIters = 0;

    // uint maxItersBtwKernelLaunches = 0;
    uint itersSinceLastKernelLaunch = 0;

    // Current GPU stream
    uint s = 0;

    // Main loop
    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000 && ! (root->state & SCORE_FINAL)) {
        if (itersSinceLastKernelLaunch >= GPU_KERNEL_LAUNCH_SPACING &&
            cudaStreamQuery(streams[s]) == cudaSuccess) {
            // Update using the last result from this stream
            if (nGpu[s]) {
                nGpu[s]->updateSim(NBLOCKS * NTHREADS-1, *winDiff[s], true);
            }
            // Get a new node to start a simulation from
            nGpu[s] = root->searchScore(root->numDescendants < MAX_NUM_NODES,
                                        useMinimax, root->state & SOLVED);
            // Put dummy result at node
            nGpu[s]->updateSim(1, 0, true);

            // Start simulation
            startNodeSimulationGpu(nGpu[s], winDiff[s], d_winDiff[s],
                d_rands[s], gen[s], turnsLeft, NBLOCKS, NTHREADS, streams[s]);
            // Uncomment to disable CPU game tree expansion:
            // cudaStreamSynchronize(streams[s]);

            #ifdef DEBUG
            err = cudaGetLastError();
            if ( cudaSuccess != err ) {
                fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                         __FILE__, __LINE__, cudaGetErrorString( err ) );
                exit( -1 );
            }
            #endif
            s = (s+1) % GPU_NUM_KERNELS;
            itersSinceLastKernelLaunch = 0;
            numIters++;
        }
        else {
            nCpu = root->searchScore(root->numDescendants < MAX_NUM_NODES,
                                    useMinimax, root->state & SOLVED);
            cpuOutcome = 0;
            for (uint i = 0; i < CPU_SIMS_PER_ITER; i++) {
                cpuOutcome += simulateNode(nCpu, moves);
            }
            nCpu->updateSim(CPU_SIMS_PER_ITER, cpuOutcome, true);
            itersSinceLastKernelLaunch++;
        }
    }

    // Finish the last GPU calls
    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        uint idx = (s+i)%GPU_NUM_KERNELS;
        cudaStreamSynchronize(streams[idx]);
        if (nGpu[idx]) {
            nGpu[idx]->updateSim(NBLOCKS * NTHREADS - 1, *winDiff[idx], true);
        }
    }

    #ifdef DEBUG
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    // Free memory
    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        curandDestroyGenerator(gen[i]);
        cudaStreamDestroy(streams[i]);
        gpuErrchk(cudaFree(d_rands[i]));
        gpuErrchk(cudaFree(d_winDiff[i]));
        gpuErrchk(cudaFreeHost(winDiff[i]));
    }

    #ifdef DEBUG
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    return numIters;
}

__global__
void cudaBlockSimulateGameKernel(Board *boards, Side side, int simsPerBoard,
    float *rands, int *winDiffs) {
    // Shared memory for partial sums
    extern __shared__ int partial_winDiff[];

    int piecesDiff;

    int numMoves;
    uint moveNum;
    uint moveIdx;

    Board b;
    Move moves[MAX_NUM_MOVES];
    Side simSide;
    int numPasses;

    // Initialize winDiffs to 0
    partial_winDiff[threadIdx.x] = 0;

    __syncthreads();

    for (uint tid = threadIdx.x; tid < simsPerBoard; tid += blockDim.x) {
        b = boards[blockIdx.x];
        simSide = side;
        moveNum = 0;
        numPasses = 0;
        while (numPasses < 2) {
            numMoves = b.getMovesAsArray(moves, simSide);
            // numMoves = b.getHeuristicMovesAsArray(moves, simSide);
            if (numMoves) {
                // rands[] is indexed by (moveNum, boardNum, tid)
                moveIdx = (uint) (rands[(moveNum * gridDim.x + blockIdx.x) * simsPerBoard + tid] * numMoves);
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
        if (threadIdx.x < s)
            partial_winDiff[threadIdx.x] += partial_winDiff[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(&winDiffs[blockIdx.x], partial_winDiff[0]);
}

void startNodeSimulationGpuBlock(Board *boards, Board *d_boards, Side side,
        uint nBoards, int *winDiffs, int *d_winDiffs,
        float *d_rands, curandGenerator_t &gen, int turnsLeft,
        uint simsPerBoard, cudaStream_t &stream) {

    // Generate random numbers for the simulations
    curandGenerateUniform(gen, d_rands, nBoards * turnsLeft * simsPerBoard);

    #ifdef DEBUG
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    // Initialize the win difference to 0
    cudaMemsetAsync(d_winDiffs, 0, nBoards * sizeof(int), stream);

    #ifdef DEBUG
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    cudaMemcpyAsync(d_boards, boards,
        nBoards * sizeof(Board), cudaMemcpyHostToDevice, stream);

    #ifdef DEBUG
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    // Run simulation kernel
    cudaBlockSimulateGameKernel<<<nBoards, simsPerBoard, simsPerBoard*sizeof(int), stream>>>(
            d_boards, side, simsPerBoard, d_rands, d_winDiffs);

    #ifdef DEBUG
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif

    // Copy result back to CPU
    cudaMemcpyAsync(winDiffs, d_winDiffs,
        nBoards * sizeof(int), cudaMemcpyDeviceToHost, stream);

    #ifdef DEBUG
    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }
    #endif
}

int expandGameTreeGpuBlock(Node *root, bool useMinimax, int ms) {
    clock_t startTime = clock();

    const int turnsLeft = root->board.countEmpty();

    cudaError err;

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    cudaStream_t streams[GPU_NUM_KERNELS];

    std::vector<std::vector<Node *>> nGpus(GPU_NUM_KERNELS);
    int *winDiffs[GPU_NUM_KERNELS];
    int *d_winDiffs[GPU_NUM_KERNELS];
    Board *boards[GPU_NUM_KERNELS];
    Board *d_boards[GPU_NUM_KERNELS];

    // Random numbers used to calculate random game states taken
    float *d_rands[GPU_NUM_KERNELS];
    curandGenerator_t gen[GPU_NUM_KERNELS];

    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        cudaStreamCreate(&streams[i]);

        curandCreateGenerator(&gen[i], CURAND_RNG_PSEUDO_DEFAULT);
        curandSetStream(gen[i], streams[i]);

        // // Allocate memory on GPU
        gpuErrchk(cudaMalloc(&d_rands[i],
            turnsLeft * MAX_NUM_MOVES * GPU_BLOCK_SIMS * sizeof(float)));
        gpuErrchk(cudaMalloc(&d_winDiffs[i], MAX_NUM_MOVES * sizeof(int)));
        gpuErrchk(cudaMallocHost(&winDiffs[i], MAX_NUM_MOVES * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_boards[i], MAX_NUM_MOVES * sizeof(Board)));
        gpuErrchk(cudaMallocHost(&boards[i], MAX_NUM_MOVES * sizeof(Board)));
    }

    uint itersSinceLastKernelLaunch = 0;

    // CPU simulation variables
    std::vector<Node *> nCpus;
    int cpuOutcome;
    Move moves[MAX_NUM_MOVES];
    int numIters = 0;

    // Current GPU stream
    uint s = 0;

    // Main loop
    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000 && !(root->state & SCORE_FINAL)) {
        if (
            itersSinceLastKernelLaunch >= GPU_KERNEL_LAUNCH_SPACING &&
            cudaStreamQuery(streams[s]) == cudaSuccess) {
            // Update using the last result from this stream
            if (nGpus[s].size() == 1) {
                nGpus[s][0]->updateSim(NBLOCKS * NTHREADS - 1, winDiffs[s][0], true);
            }
            else {
                for (uint i = 0; i < nGpus[s].size(); i++) {
                    nGpus[s][i]->updateSim(GPU_BLOCK_SIMS - 1, winDiffs[s][i],
                        true);
                }
            }
            // Get a new node to start a simulation from
            nGpus[s] = root->searchScoreBlock(
                root->numDescendants < MAX_NUM_NODES,
                useMinimax, root->state & SOLVED);
            // Put dummy result at node
            for (Node *n : nGpus[s]) {
                n->updateSim(1, 0, true);
            }

            for (uint i = 0; i < nGpus[s].size(); i++) {
                boards[s][i] = nGpus[s][i]->board;
            }

            // Start simulation
            if (nGpus[s].size() > 1) {
                startNodeSimulationGpuBlock(boards[s], d_boards[s], nGpus[s][0]->side,
                    nGpus[s].size(), winDiffs[s], d_winDiffs[s],
                    d_rands[s], gen[s], turnsLeft, GPU_BLOCK_SIMS, streams[s]);
            }
            else {
                startNodeSimulationGpu(nGpus[s][0], &winDiffs[s][0], &d_winDiffs[s][0],
                    d_rands[s], gen[s], turnsLeft, NBLOCKS, NTHREADS, streams[s]);
            }

            // Uncomment to disable CPU game tree expansion:
            // cudaStreamSynchronize(streams[s]);

            #ifdef DEBUG
            err = cudaGetLastError();
            if ( cudaSuccess != err ) {
                fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                         __FILE__, __LINE__, cudaGetErrorString( err ) );
                exit( -1 );
            }
            #endif
            numIters += nGpus[s].size();
            s = (s+1) % GPU_NUM_KERNELS;
            itersSinceLastKernelLaunch = 0;
        }
        else {
            nCpus = root->searchScoreBlock(root->numDescendants < MAX_NUM_NODES,
                                    useMinimax, root->state & SOLVED);
            for (Node *n : nCpus) {
                cpuOutcome = 0;
                for (uint i = 0; i < CPU_BLOCK_SIMS; i++) {
                    cpuOutcome += simulateNode(n, moves);
                }
                n->updateSim(CPU_BLOCK_SIMS, cpuOutcome, true);
            }
            itersSinceLastKernelLaunch++;
        }
    }

    // Finish the last GPU calls
    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        uint idx = (s+i)%GPU_NUM_KERNELS;
        cudaStreamSynchronize(streams[idx]);
        if (nGpus[idx].size() == 1) {
            nGpus[idx][0]->updateSim(NBLOCKS * NTHREADS - 1, winDiffs[idx][0], true);
        }
        else {
            for (uint i = 0; i < nGpus[idx].size(); i++) {
                nGpus[idx][i]->updateSim(GPU_BLOCK_SIMS - 1, winDiffs[idx][i], true);
            }
        }
    }

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // Free memory
    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        curandDestroyGenerator(gen[i]);
        cudaStreamDestroy(streams[i]);
        gpuErrchk(cudaFree(d_rands[i]));
        gpuErrchk(cudaFree(d_winDiffs[i]));
        gpuErrchk(cudaFreeHost(winDiffs[i]));
    }

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return numIters;
}
