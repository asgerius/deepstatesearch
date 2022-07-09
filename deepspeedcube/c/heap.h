#ifndef HEAPH
#define HEAPH

#include <stdlib.h>
#include <string.h>

#define HEAP_PARENT(x) ((x) / 2)
#define HEAP_LEFT(x) (2 * (x))
#define HEAP_RIGHT(x) (2 * (x) + 1)

/* Alg
heap structure
struct heap_entry {
    float value,
    char *state
}
heap of states
size_t stride;
char *states_heap[n][stride];

int get_best_states() {

    for i in points to extract do
        value, ptr = extract_min(heap, states_heap)
        if state not seen do
            append state to new_states
        fi
    end do

    return len(new_states)
}

void insert_neighbours() {

    for state in neighbours(new_states) do
        insert(heap, f(state), states_heap, f(state))
    end do
}
 */

struct heap {
    float *keys;
    void *data;
    size_t element_size;
    size_t num_elems;       // Includes null element
    size_t allocated_size;  // Number of allocated elements including null element
};

struct heap_entry {
    float key;
    void *data;
};

typedef struct heap heap;
typedef struct heap_entry heap_entry;

heap_entry heap_min(heap heap) {
    heap_entry entry = {
        .key = heap.keys[1],
        .data = heap.data + heap.element_size,
    };
    return entry;
}

heap_entry heap_extract_min(heap heap);

void heap_insert(heap heap, heap_entry entry);

void heap_decrease_key(heap heap, size_t index, float key);

#endif
