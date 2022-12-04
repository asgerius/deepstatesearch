CC      = gcc
CFLAGS  = -g0 -O3 -funroll-loops -fopenmp -ffast-math -funsigned-char
CSHARED = -shared -fPIC

CPP     = g++
CXX	    = nvcc

OPT     = --compiler-options "-g0 -O3 -funroll-loops"
XOPTS   = -Xptxas=-v
ARCH    = -arch=sm_70
OMP     = -fopenmp

CUDA_PATH = /appl/cuda/11.5.1

all:
	make clean
	mkdir -p lib/envs
	make lib/libdss.so
	@if command -v nvcc ; then\
		make lib/cube_cuda.so;\
	fi

lib/libdss.so:
	$(CC) -o $@ \
		deepstatesearch/c/hashmap.c/hashmap.c deepstatesearch/c/values.c \
		deepstatesearch/c/astar.c deepstatesearch/c/heap.c deepstatesearch/c/unique.c \
		deepstatesearch/c/envs/envs.c deepstatesearch/c/envs/cube.c deepstatesearch/c/envs/sliding.c \
		$(CFLAGS) $(CSHARED)

lib/cube_cuda.so:
	$(CXX) $(OPT) $(ARCH) $(XOPTS) -Xcompiler "-fPIC" -dc \
		deepstatesearch/cuda/envs/cube.cu \
		-o lib/cube_cuda.o
	$(CXX) $(ARCH) $(XOPTS) -Xcompiler "-fPIC" -dlink \
		lib/cube_cuda.o \
		-o lib/link.o
	$(CPP) -shared -L$(CUDA_PATH)/lib64 -lcudart \
		lib/cube_cuda.o lib/link.o \
		-o lib/cube_cuda.so

clean:
	$(RM) -r lib/*
