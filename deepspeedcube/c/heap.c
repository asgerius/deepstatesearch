#include "heap.h"


heap *heap_alloc(size_t element_size) {
    heap *heap_p = malloc(sizeof(*heap_p));
    heap_p->entries = malloc(HEAP_BASE_SIZE * sizeof(heap_entry));
    heap_p->element_size = element_size;
    heap_p->num_elems = 1;
    heap_p->num_alloc = HEAP_BASE_SIZE;

    return heap_p;
}

void heap_free(heap *heap_p) {
    free(heap_p->entries);
    free(heap_p);
}

int heap_should_increase_alloc(heap *heap_p, size_t new_elems) {
    return heap_p->num_elems + new_elems > heap_p->num_alloc;
}

void heap_increase_alloc(heap *heap_p) {

    // Create new array for entries
    heap_entry *new_entry_array = malloc(2 * heap_p->num_alloc * sizeof(heap_entry));
    #pragma omp parallel for
    for (size_t i = 0; i < heap_p->num_alloc; ++ i) {
        new_entry_array[i] = heap_p->entries[i];
    }
    free(heap_p->entries);
    heap_p->entries = new_entry_array;

    heap_p->num_alloc *= 2;
}

void bubble_up(heap *heap_p, size_t index) {
    if (index == 1) {
        return;
    }

    size_t index_parent = HEAP_PARENT(index);

    float key_node = heap_p->entries[index].key;
    float key_parent = heap_p->entries[index_parent].key;

    while (key_node < key_parent && index > 1) {
        // Heap order violated, so swap elements
        heap_entry entry_tmp = heap_p->entries[index];
        heap_p->entries[index] = heap_p->entries[index_parent];
        heap_p->entries[index_parent] = entry_tmp;

        index = index_parent;
        index_parent = HEAP_PARENT(index);

        key_node = heap_p->entries[index].key;
        key_parent = heap_p->entries[index_parent].key;
    }
}

void bubble_down(heap *heap_p, size_t index) {

    size_t index_left = HEAP_LEFT(index);
    size_t index_right = HEAP_RIGHT(index);

    if (index_left > heap_p->num_elems) {
        // No children, so stop here
        return;
    }
    // Only the left child exist, so make both children the same to
    // make all following logic work without the need for exceptions
    index_right = EITHER_OR(index_right > heap_p->num_elems, index_left, index_right);

    float key_node = heap_p->entries[index].key;
    float key_left = heap_p->entries[index_left].key;
    float key_right = heap_p->entries[index_right].key;

    bool key_left_smaller = key_left < key_right;
    float key_child = EITHER_OR(key_left_smaller, key_left, key_right);
    size_t index_child = EITHER_OR(key_left_smaller, index_left, index_right);

    while (key_node > key_child && index < heap_p->num_elems - 1) {
        // Heap order violated, so swap elements
        heap_entry entry_tmp = heap_p->entries[index];
        heap_p->entries[index] = heap_p->entries[index_child];
        heap_p->entries[index_child] = entry_tmp;

        index = index_child;
        index_left = HEAP_LEFT(index);
        index_right = HEAP_RIGHT(index);

        if (index_left > heap_p->num_elems) {
            // No children, so stop here
            return;
        }
        index_right = EITHER_OR(index_right > heap_p->num_elems, index_left, index_right);

        key_node = heap_p->entries[index].key;
        key_left = heap_p->entries[index_left].key;
        key_right = heap_p->entries[index_right].key;

        key_left_smaller = key_left < key_right;
        key_child = EITHER_OR(key_left_smaller, key_left, key_right);
        index_child = EITHER_OR(key_left_smaller, index_left, index_right);
    }
}

size_t heap_extract_min(heap *heap_p, size_t n, float *keys, void *data) {
    // Extracts up to n elements from the heap
    // Returns the exact number of elements extracted
    n = MIN(n, heap_p->num_elems - 1);

    for (size_t i = 0; i < n; ++ i) {
        // Copy key and data to out arrays
        void *data_out = data + i * heap_p->element_size;
        memcpy(data_out, heap_p->entries[1].data, heap_p->element_size);

        // Set first element to element after the last and update heap order
        -- heap_p->num_elems;
        heap_entry entry_tmp = heap_p->entries[1];
        heap_p->entries[1] = heap_p->entries[heap_p->num_elems];
        heap_p->entries[heap_p->num_elems] = entry_tmp;

        bubble_down(heap_p, 1);
    }

    return n;
}

void heap_insert(heap *heap_p, size_t n, const float *keys, const void *data) {
    /* Insert n elements into the heap. This function assumes that enough data has
    been allocated to the heap, which can be checked with heap_should_increase_alloc.
    If not, use heap_increase_alloc to double the amount of available memory. */

    for (size_t i = 0; i < n; ++ i) {
        heap_p->entries[heap_p->num_elems].key = keys[i];
        heap_p->entries[heap_p->num_elems].data = data;

        bubble_up(heap_p, heap_p->num_elems);

        ++ heap_p->num_elems;
    }
}

void heap_decrease_key(heap *heap_p, size_t index, float key) {
    heap_p->entries[index].key = key;

    bubble_up(heap_p, index);
}
