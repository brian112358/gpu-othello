CC              = /usr/bin/g++
NVCC            = $(CUDA_BIN_PATH)/nvcc

CUDA_PATH       = /usr/local/cuda
CUDA_INC_PATH   = $(CUDA_PATH)/include
CUDA_BIN_PATH   = $(CUDA_PATH)/bin
CUDA_LIB_PATH   = $(CUDA_PATH)/lib

# CUDA code generation flags
GENCODE_FLAGS = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52

CUDA_LIB_PATH = $(CUDA_LIB_PATH)64
LDFLAGS       = -L$(CUDA_LIB_PATH) -lcufft -lcudart -lcurand
CCFLAGS       = -std=c++11 -m64 -Wall -pedantic -O3
NVCCFLAGS     = -m64 -lcufft -lcurand

OBJS        = board.o simulate.o gametree.o player.o
PLAYERNAME  = gpu-othello

all: $(PLAYERNAME) testgame

$(PLAYERNAME): $(OBJS) wrapper.o
	$(CC) $(CCFLAGS) $^ -o $@ -I$(CUDA_INC_PATH)

testgame: testgame.o
	$(CC) $(CCFLAGS) $^ -o $@

test: $(OBJS) test.cpp
	$(CC) $(CCFLAGS) $^ -o $@

testgame.o: testgame.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

wrapper.o: wrapper.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

player.o: player.cpp player.hpp
	$(CC) $(CCFLAGS) -c $< -o $@

board.o: board.cpp board.hpp
	$(CC) $(CCFLAGS) -c $< -o $@

gametree.o: gametree.cpp gametree.hpp
	$(CC) $(CCFLAGS) -c $< -o $@

simulate.o: simulate.cpp simulate.hpp
	$(CC) $(CCFLAGS) -c $< -o $@

java:
	make -C java/

cleanjava:
	make -C java/ clean

clean:
	rm -f *.o $(PLAYERNAME) test testgame

.PHONY: java
