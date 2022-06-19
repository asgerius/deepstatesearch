#ifndef UNIQUEH
#define UNIQUEH

#include "astar.h"
#include "hashmap.c/hashmap.h"

int unique(
    void *array,
    size_t n,
    size_t stride,
    unsigned long *index,
    unsigned long *inverse
);

#endif
