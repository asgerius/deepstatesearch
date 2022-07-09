#ifndef ASTARH
#define ASTARH

#include <stdlib.h>
#include <string.h>
#include "hashmap.c/hashmap.h"

typedef struct hashmap hashmap;

struct hm_entry {
    void *p_elem;
    size_t bytes;
};

typedef struct hm_entry hm_entry;

uint64_t hash(const void *elem, uint64_t seed0, uint64_t seed1);

int compare(const void *elem1, const void *elem2, void *udata);

hashmap *new_hashmap();

void already_exist(
    size_t n,
    size_t stride,
    bool *seen,
    void *new_states,
    hashmap *seen_states
);

#endif
