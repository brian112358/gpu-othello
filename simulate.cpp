#include "simulate.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "board.hpp"
#include "common.hpp"

#include <cuda_runtime.h>
#include <curand.h>

// Experimentally verified to stay within 768MB memory limit
#define MAX_NUM_NODES 500000


#define CPU_SIMS_PER_ITER 1

#define GPU_KERNEL_LAUNCH_SPACING 16
#define GPU_NUM_KERNELS 16
// #define GPU_SIMS_PER_ITER 1024
#define NBLOCKS 8
#define NTHREADS 128

#define CPU_BLOCK_SIMS_PER_ITER 1
#define GPU_BLOCK_SIMS_PER_ITER 1024

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
        n = root->searchScore(root->numDescendants < MAX_NUM_NODES, useMinimax);
        outcome = simulateNode(n, moves);
        n->updateSim(1, outcome);
        numSims++;
    }
    return numSims;
}


__global__
void cudaSimulateGameKernel(Board board, Side side, int simsPerBlock,
    float *rands, int *winDiff) {
    // Shared memory for partial sums
    extern __shared__ int partial_winDiff[];

    const uint tid = threadIdx.x;

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

    b = board;
    simSide = side;
    moveNum = 0;
    numPasses = 0;
    while (numPasses < 2) {
        numMoves = b.getMovesAsArray(moves, simSide);
        if (numMoves) {
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

void startNodeSimulationGpu(Node *n, int *winDiff, int *d_winDiff,
        float *d_rands, curandGenerator_t &gen, int turnsLeft,
        uint nBlocks, uint nThreads, cudaStream_t stream) {

    // Generate random numbers for the simulations
    curandGenerateUniform(gen, d_rands, turnsLeft * nBlocks * nThreads);

    // Initialize the win difference to 0
    cudaMemset(d_winDiff, 0, 1 * sizeof(int));

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

    cudaError err;

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

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
    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000) {
        if (itersSinceLastKernelLaunch >= GPU_KERNEL_LAUNCH_SPACING &&
            cudaStreamQuery(streams[s]) == cudaSuccess) {
            // if (itersSinceLastKernelLaunch > maxItersBtwKernelLaunches)
            //     maxItersBtwKernelLaunches = itersSinceLastKernelLaunch;
            // Update using the last result from this stream
            if (nGpu[s]) {
                nGpu[s]->updateSim(NBLOCKS * NTHREADS-1, *winDiff[s]);
            }
            // Get a new node to start a simulation from
            nGpu[s] = root->searchScore(root->numDescendants < MAX_NUM_NODES,
                                        useMinimax);
            // Put dummy result at node
            nGpu[s]->updateSim(1, 0);
            
            // Start simulation
            startNodeSimulationGpu(nGpu[s], winDiff[s], d_winDiff[s],
                d_rands[s], gen[s], turnsLeft, NBLOCKS, NTHREADS, streams[s]);
            // cudaStreamSynchronize(streams[s]);

            err = cudaGetLastError();
            if ( cudaSuccess != err ) {
                fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                         __FILE__, __LINE__, cudaGetErrorString( err ) );
                exit( -1 );
            }
            s = (s+1) % GPU_NUM_KERNELS;
            itersSinceLastKernelLaunch = 0;
        }
        else {
            nCpu = root->searchScore(root->numDescendants < MAX_NUM_NODES,
                                    useMinimax);
            cpuOutcome = 0;
            for (uint i = 0; i < CPU_SIMS_PER_ITER; i++) {
                cpuOutcome += simulateNode(nCpu, moves);
            }
            nCpu->updateSim(CPU_SIMS_PER_ITER, cpuOutcome);
            itersSinceLastKernelLaunch++;
        }

        numIters++;
    }

    // Finish the last GPU calls
    for (uint i = 0; i < GPU_NUM_KERNELS; i++) {
        uint idx = (s+i)%GPU_NUM_KERNELS;
        cudaStreamSynchronize(streams[idx]);
        if (nGpu[idx]) {
            nGpu[idx]->updateSim(NBLOCKS * NTHREADS-1, *winDiff[idx]);
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
        gpuErrchk(cudaFree(d_winDiff[i]));
        gpuErrchk(cudaFreeHost(winDiff[i]));
    }

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return numIters;
}

__global__
void cudaBlockSimulateGameKernel(Board *boards, Side side, int simsPerBoard,
    float *rands, int *winDiffs) {
    // Shared memory for partial sums
    extern __shared__ int partial_winDiff[];

    const uint tid = threadIdx.x;

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

    b = boards[blockIdx.x];
    simSide = side;
    moveNum = 0;
    numPasses = 0;
    while (numPasses < 2) {
        numMoves = b.getMovesAsArray(moves, simSide);
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

    // Reduction step
    __syncthreads();
    for (uint s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            partial_winDiff[tid] += partial_winDiff[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(&winDiffs[blockIdx.x], partial_winDiff[0]);
}

void startNodeSimulationGpuBlock(Board *boards, Board *d_boards, Side side,
        uint nBoards, int *winDiffs, int *d_winDiffs,
        float *d_rands, curandGenerator_t &gen, int turnsLeft,
        uint simsPerBoard, cudaStream_t stream) {

    // Generate random numbers for the simulations
    curandGenerateUniform(gen, d_rands, nBoards * turnsLeft * simsPerBoard);

    // Initialize the win difference to 0
    cudaMemset(d_winDiffs, 0, nBoards * sizeof(int));

    cudaMemcpyAsync(d_boards, boards,
        nBoards * sizeof(Board), cudaMemcpyHostToDevice, stream);

    // Run simulation kernel
    cudaBlockSimulateGameKernel<<<nBoards, simsPerBoard, simsPerBoard*sizeof(int), stream>>>(
            d_boards, side, simsPerBoard, d_rands, d_winDiffs);

    // Copy result back to CPU
    cudaMemcpyAsync(winDiffs, d_winDiffs,
        nBoards * sizeof(int), cudaMemcpyDeviceToHost, stream);
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

    cudaStream_t stream;

    std::vector<Node *> nGpus;
    int *winDiffs;
    int *d_winDiffs;
    Board *boards;
    Board *d_boards;

    // Random numbers used to calculate random game states taken
    float *d_rands;
    curandGenerator_t gen;

    cudaStreamCreate(&stream);

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(gen, stream);

    // // Allocate memory on GPU
    gpuErrchk(cudaMalloc(&d_rands,
        turnsLeft * MAX_NUM_MOVES * GPU_BLOCK_SIMS_PER_ITER * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_winDiffs, MAX_NUM_MOVES * sizeof(int)));
    gpuErrchk(cudaMallocHost(&winDiffs, MAX_NUM_MOVES * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_boards, MAX_NUM_MOVES * sizeof(Board)));
    gpuErrchk(cudaMallocHost(&boards, MAX_NUM_MOVES * sizeof(Board)));
    
    // CPU simulation variables
    std::vector<Node *> nCpus;
    int cpuOutcome;
    Move moves[MAX_NUM_MOVES];
    int numIters = 0;
    
    // Main loop
    while (clock() - startTime < ms * CLOCKS_PER_SEC / 1000) {
        if (cudaStreamQuery(stream) == cudaSuccess) {
            // Update using the last result from this stream
            if (nGpus.size() == 1) {
                nGpus[0]->updateSim(NBLOCKS * NTHREADS - 1, winDiffs[0]);
            }
            else {
                for (Node *n : nGpus) {
                    n->updateSim(GPU_BLOCK_SIMS_PER_ITER-1, *winDiffs);
                }
            }
            // Get a new node to start a simulation from
            nGpus = root->searchScoreBlock(root->numDescendants < MAX_NUM_NODES,
                                        useMinimax);
            // Put dummy result at node
            for (Node *n : nGpus) {
                n->updateSim(1, 0);
            }

            for (uint i = 0; i < nGpus.size(); i++) {
                boards[i] = nGpus[i]->board;
            }
            
            // Start simulation
            if (nGpus.size() > 1) {
                startNodeSimulationGpuBlock(boards, d_boards, nGpus[0]->side,
                    nGpus.size(), winDiffs, d_winDiffs,
                    d_rands, gen, turnsLeft, GPU_BLOCK_SIMS_PER_ITER, stream);   
            }
            else {
                startNodeSimulationGpu(nGpus[0], &winDiffs[0], &d_winDiffs[0],
                    d_rands, gen, turnsLeft, NBLOCKS, NTHREADS, stream);
            }
            // cudaStreamSynchronize(stream);

            err = cudaGetLastError();
            if ( cudaSuccess != err ) {
                fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                         __FILE__, __LINE__, cudaGetErrorString( err ) );
                exit( -1 );
            }
        }
        else {
            nCpus = root->searchScoreBlock(root->numDescendants < MAX_NUM_NODES,
                                    useMinimax);
            for (Node *n : nCpus) {
                cpuOutcome = 0;
                for (uint i = 0; i < CPU_BLOCK_SIMS_PER_ITER; i++) {
                    cpuOutcome += simulateNode(n, moves);
                }
                n->updateSim(CPU_BLOCK_SIMS_PER_ITER, cpuOutcome);
            }
        }

        numIters++;
    }

    // Finish the last GPU call
    cudaStreamSynchronize(stream);
    for (uint i = 0; i < nGpus.size(); i++) {
        nGpus[i]->updateSim(GPU_BLOCK_SIMS_PER_ITER-1, winDiffs[i]);
    }

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // Free memory
    curandDestroyGenerator(gen);
    cudaStreamDestroy(stream);
    gpuErrchk(cudaFree(d_rands));
    gpuErrchk(cudaFree(d_winDiffs));
    gpuErrchk(cudaFreeHost(winDiffs));

    err = cudaGetLastError();
    if ( cudaSuccess != err ) {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 __FILE__, __LINE__, cudaGetErrorString( err ) );
        exit( -1 );
    }

    return numIters;
}