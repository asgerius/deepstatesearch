#include "heap.h"


heap *heap_alloc(size_t d, size_t element_size) {
    heap *heap_p = malloc(sizeof(*heap_p));
    heap_p->entries = malloc(HEAP_BASE_SIZE * sizeof(heap_entry));
    heap_p->d = d;
    heap_p->element_size = element_size;
    heap_p->num_elems = 0;
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
    if (index == 0) {
        return;
    }

    size_t index_parent = HEAP_PARENT(index, heap_p->d);

    float key_node = heap_p->entries[index].key;
    float key_parent = heap_p->entries[index_parent].key;

    if (key_parent > key_node) {
        // Heap order violated, so swap elements
        heap_entry entry_tmp = heap_p->entries[index];
        heap_p->entries[index] = heap_p->entries[index_parent];
        heap_p->entries[index_parent] = entry_tmp;

        bubble_up(heap_p, index_parent);
    }
}

void bubble_down(heap *heap_p, size_t index) {

    size_t index_start_child = HEAP_LEFTMOST_CHILD(index, heap_p->d);
    if (index_start_child >= heap_p->num_elems) {
        // No children, so stop here
        return;
    }

    float key_node = heap_p->entries[index].key;

    size_t index_lowest_child = index_start_child;
    float key_lowest_child = heap_p->entries[index_lowest_child].key;

    #pragma unroll
    for (size_t i = 1; i < heap_p->d; ++ i) {
        size_t index_child = index_start_child + i;
        index_child = EITHER_OR(index_child < heap_p->num_elems, index_child, heap_p->num_elems - 1);
        float key_child = heap_p->entries[index_child].key;
        bool new_smallest = key_child < key_lowest_child;
        key_lowest_child = EITHER_OR(new_smallest, key_child, key_lowest_child);
        index_lowest_child = EITHER_OR(new_smallest, index_child, index_lowest_child);
    }

    if (key_node > key_lowest_child) {
        // Heap order violated, so swap elements
        heap_entry entry_tmp = heap_p->entries[index];
        heap_p->entries[index] = heap_p->entries[index_lowest_child];
        heap_p->entries[index_lowest_child] = entry_tmp;

        bubble_down(heap_p, index_lowest_child);
    }
}

size_t heap_extract_min(heap *heap_p, size_t n, float *keys, void *data) {
    // Extracts up to n elements from the heap
    // Returns the exact number of elements extracted
    n = MIN(n, heap_p->num_elems);

    for (size_t i = 0; i < n; ++ i) {
        // Copy key and data to out arrays
        void *data_out = data + i * heap_p->element_size;
        memcpy(data_out, heap_p->entries[0].data, heap_p->element_size);

        // Swap first and last elements and update heap order
        -- heap_p->num_elems;
        heap_entry entry_tmp = heap_p->entries[0];
        heap_p->entries[0] = heap_p->entries[heap_p->num_elems];
        heap_p->entries[heap_p->num_elems] = entry_tmp;

        bubble_down(heap_p, 0);
    }

    return n;
}

void heap_insert(heap *heap_p, float key, const void *data) {
    /* Insert n elements into the heap. This function assumes that enough data has
    been allocated to the heap, which can be checked with heap_should_increase_alloc.
    If not, use heap_increase_alloc to double the amount of available memory. */

    heap_p->entries[heap_p->num_elems].key = key;
    heap_p->entries[heap_p->num_elems].data = data;

    bubble_up(heap_p, heap_p->num_elems);

    ++ heap_p->num_elems;
}

void heap_decrease_key(heap *heap_p, size_t index, float key) {
    heap_p->entries[index].key = key;

    bubble_up(heap_p, index);
}
