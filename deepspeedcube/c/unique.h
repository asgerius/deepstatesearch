#ifndef UNIQUEH
#define UNIQUEH

#include <stdlib.h>
#include <string.h>

#include "astar.h"
#include "hashmap.c/hashmap.h"


struct hm_entry {
    void *p_elem;
    size_t bytes;
};

typedef struct hashmap hashmap;
typedef struct hm_entry hm_entry;

int unique(
    void *array,
    size_t n,
    size_t stride,
    unsigned long *index,
    unsigned long *inverse
);

#endif
