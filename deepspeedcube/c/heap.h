#ifndef HEAP_H
#define HEAP_H

#include <stdlib.h>
#include <string.h>

#include "envs/envs.h"

#define HEAP_BASE_SIZE 20000
#define HEAP_PARENT(x) ((x) / 2)
#define HEAP_LEFT(x) (2 * (x))
#define HEAP_RIGHT(x) (2 * (x) + 1)


struct heap_entry {
    float key;
    const void *data;
};

typedef struct heap_entry heap_entry;

struct heap {
    heap_entry *entries;
    size_t element_size;
    // Number of elements (including null element) in the heap
    // This will always be less than or equal to the number of elements
    // that memory has been allocated for
    size_t num_elems;
    size_t num_alloc;
};

typedef struct heap heap;

/* Allocates room for an empty heap. */
heap *heap_alloc(size_t element_size);

/* Frees the heap. The arrays are owned by numpy/torch, so they are not freed here. */
void heap_free(heap *heap_p);

int heap_should_increase_alloc(heap *heap_p, size_t new_elems);

/* Allocates more memory to the heap. */
void heap_increase_alloc(heap *heap_p);

/* Extracts the n lowest keys in the heap and puts the resulting keys and data in the
given arrays in order of ascending keys. */
size_t heap_extract_min(heap *heap_p, size_t n, float *keys, void *data);

void heap_insert(heap *heap_p, size_t n, const float *keys, const void *data);

void heap_decrease_key(heap *heap_p, size_t index, float key);

#endif
