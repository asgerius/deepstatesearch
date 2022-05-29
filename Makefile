CC=gcc
CFLAGS=-g -O3 -funroll-loops -fopenmp -shared -fPIC
LDFLAGS=

lib/cube.so:
	$(CC) -o $@ deepspeedcube/c/envs/cube.c $(CFLAGS) $(LDFLAGS)

clean:
	$(RM) -r cube *.o
