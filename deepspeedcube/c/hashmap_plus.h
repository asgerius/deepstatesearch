/* Utilities hashmap.c */
#ifndef HASHMAPPLUSH
#define HASHMAPPLUSH

#include <stdlib.h>
#include <string.h>
#include "hashmap.c/hashmap.h"

struct hm_entry {
    void *p_elem;
    size_t bytes;
};

typedef struct hashmap hashmap;
typedef struct hm_entry hm_entry;

uint64_t hash(const void *elem, uint64_t seed0, uint64_t seed1);

int compare(const void *elem1, const void *elem2, void *udata);

hashmap *create_hashmap();

#endif
