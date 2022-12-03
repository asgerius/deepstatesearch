#ifndef VALUES_H
#define VALUES_H

#include <stdlib.h>

#include "envs/envs.h"
#include "hashmap.c/hashmap.h"


struct node {
    float J;
    size_t state_size;
    void *state;
    uint64_t hash;
};

typedef struct hashmap hashmap;
typedef struct node node;

hashmap *values_node_map_from_states(size_t n, size_t state_size, void *states, action action_space_size);

void values_free(hashmap *node_map);

void values_set(size_t n, size_t state_size, void *states, float *J, hashmap *node_map);

#endif
