CC=gcc
CFLAGS=-g -O3 -funroll-loops -fopenmp -shared -fPIC
LDFLAGS=

all:
	make clean
	mkdir -p lib
	make lib/cube.so

lib/cube.so:
	$(CC) -o $@ deepspeedcube/c/envs/cube.c $(CFLAGS) $(LDFLAGS)

clean:
	$(RM) -r lib/*
