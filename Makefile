CC     = gcc
CFLAGS = -g0 -O3 -funroll-loops -fopenmp -shared -fPIC

CPP    = g++
CXX	   = nvcc

OPT    = --compiler-options "-g0 -O3 -funroll-loops"
XOPTS  = -Xptxas=-v
ARCH   = -arch=sm_70
OMP    = -fopenmp

CUDA_PATH ?= /appl/cuda/11.5.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

all:
	make clean
	mkdir -p lib
	make lib/cube.so
	make lib/cube_cuda.so

lib/cube.so:
	$(CC) -o $@ deepspeedcube/c/envs/cube.c $(CFLAGS)

lib/cube_cuda.so:
	$(CXX) $(OPT) $(ARCH) $(XOPTS) -Xcompiler "-fPIC" -dc \
		deepspeedcube/cuda/envs/cube.cu \
		-o lib/cube_cuda.o
	$(CXX) $(ARCH) $(XOPTS) -Xcompiler "-fPIC" -dlink \
		lib/cube_cuda.o \
		-o lib/link.o
	$(CPP) -shared -L/appl/cuda/11.5.1/lib64 -lcudart \
		lib/cube_cuda.o lib/link.o \
		-o lib/cube_cuda.so

clean:
	$(RM) -r lib/*
