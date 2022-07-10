#ifndef ASTARH
#define ASTARH

#include <stdlib.h>
#include <string.h>
#include "hashmap_plus.h"
#include "hashmap.c/hashmap.h"


void already_exist(
    size_t n,
    size_t stride,
    bool *seen,
    void *new_states,
    hashmap *seen_states
);

#endif
