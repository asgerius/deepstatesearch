#ifndef HEAPH
#define HEAPH

#include <stdlib.h>
#include <string.h>

#define HEAP_PARENT(x) ((x) / 2)
#define HEAP_LEFT(x) (2 * (x))
#define HEAP_RIGHT(x) (2 * (x) + 1)


struct heap {
    float *keys;
    void *data;
    size_t element_size;
    // Number of elements (excluding null element) in the heap
    // This will always be less than or equal to the number of elements
    // that memory has been allocated for
    size_t num_elems;
};

typedef struct heap heap;

/* Allocates room for an empty heap. */
heap *heap_alloc(float *keys, void *data, size_t element_size);

/* Frees the heap. The arrays are owned by numpy/torch, so they are not freed here. */
void heap_free(heap *heap);

/* Updates the key and data pointers in the heap. Useful when changing allocated memory. */
void heap_update_ptrs(heap *heap, float *keys, void *data);

/* Extracts the n lowest keys in the heap and puts the resulting keys and data in the
given arrays in order of ascending keys. */
void heap_extract_min(heap *heap, size_t n, float *keys, void *data);

void heap_insert(heap *heap, size_t n, float *keys, const void *data);

void heap_decrease_key(heap *heap, size_t index, float key);

#endif
