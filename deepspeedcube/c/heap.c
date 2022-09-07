#include "heap.h"
#include <stdio.h>

heap *heap_alloc(size_t element_size) {
    heap *heap_p = malloc(sizeof(*heap_p));
    heap_p->entries = malloc(HEAP_BASE_SIZE * sizeof(heap_entry));
    heap_p->element_size = element_size;
    heap_p->num_elems = 1;
    heap_p->num_alloc = HEAP_BASE_SIZE;
    heap_p->num_data_arrays = 1;
    heap_p->data = malloc(sizeof(void *));
    heap_p->data[0] = malloc(HEAP_BASE_SIZE * element_size * heap_p->num_alloc);

    heap_p->entries[0].data = heap_p->data[0];

    #pragma omp parallel for
    for (size_t i = 1; i < heap_p->num_alloc; ++ i) {
        heap_p->entries[i].data = heap_p->entries[0].data + i * element_size;
    }

    return heap_p;
}

void heap_free(heap *heap_p) {
    for (size_t i = 0; i < heap_p->num_data_arrays; ++ i) {
        free(heap_p->data[i]);
    }
    free(heap_p->entries);
    free(heap_p);
}

int heap_should_increase_alloc(heap *heap_p, size_t new_elems) {
    // printf("Increase alloc with %zu + %zu > %zu?\n", heap_p->num_elems, new_elems, heap_p->num_alloc);
    return heap_p->num_elems + new_elems > heap_p->num_alloc;
}

void heap_increase_alloc(heap *heap_p) {
    // printf("Doubling allocation\n");

    // Create new array for entries

    // for (size_t i = 0; i < heap_p->num_data_arrays; ++ i) {
    //     printf("Data pointer %zu = %p\n", i, heap_p->data[i]);
    // }
    // for (size_t i = 0; i < heap_p->num_alloc; ++ i) {
    //     printf("Entry %zu = %p\n", i, heap_p->entries[i].data);
    // }
    heap_entry *new_entry_array = malloc(2 * heap_p->num_alloc * sizeof(heap_entry));
    memcpy(
        new_entry_array,
        heap_p->entries,
        heap_p->num_alloc * sizeof(heap_entry)
    );
    // #pragma omp parallel for
    // for (size_t i = 0; i < heap_p->num_elems; ++ i) {
        // new_entry_array[i] = heap_p->entries[i];
    // }
    free(heap_p->entries);
    heap_p->entries = new_entry_array;

    // for (size_t i = 0; i < heap_p->num_data_arrays; ++ i) {
    //     printf("Data pointer %zu = %p\n", i, heap_p->data[i]);
    // }
    // for (size_t i = 0; i < heap_p->num_alloc; ++ i) {
    //     printf("Entry %zu = %p\n", i, new_entry_array[i].data);
    // }

    // printf("Creating new data array\n");

    // Create new data array
    void **new_data_arrays = malloc((heap_p->num_data_arrays + 1) * sizeof(void *));
    // printf("Copying %zu bytes to new data array\n", heap_p->num_data_arrays * sizeof(void *));
    for (size_t i = 0; i < heap_p->num_data_arrays; ++ i) {
        // printf("Copying data pointer at %p\n", heap_p->data[i]);
        new_data_arrays[i] = heap_p->data[i];
    }
    free(heap_p->data);
    heap_p->data = new_data_arrays;
    // printf("Set new data arrays\n");
    void *new_data = malloc(heap_p->num_alloc * heap_p->element_size);
    heap_p->data[heap_p->num_data_arrays] = new_data;
    heap_p->entries[heap_p->num_alloc].data = new_data;

    // for (size_t i = 0; i < heap_p->num_data_arrays + 1; ++ i) {
    //     printf("Pointer %zu = %p\n", i, heap_p->data[i]);
    // }

    // Set pointers in the new entries to the new array
    #pragma omp parallel for
    for (size_t i = heap_p->num_alloc + 1; i < 2 * heap_p->num_alloc; ++ i) {
        heap_p->entries[i].data = new_data + (i - heap_p->num_alloc) * heap_p->element_size;
    }

    heap_p->num_alloc *= 2;
    ++ heap_p->num_data_arrays;

    // for (size_t i = 0; i < heap_p->num_data_arrays; ++ i) {
    //     printf("Data pointer %zu = %p\n", i, heap_p->data[i]);
    // }
    // for (size_t i = 0; i < heap_p->num_alloc; ++ i) {
    //     printf("Entry %zu = %p\n", i, heap_p->entries[i].data);
    // }
}

void bubble_up(heap *heap_p, size_t index) {
    if (index == 1) {
        return;
    }

    size_t index_parent = HEAP_PARENT(index);

    float key_node = heap_p->entries[index].key;
    float key_parent = heap_p->entries[index_parent].key;

    if (key_node < key_parent) {
        // Heap order violated, so swap elements
        heap_entry entry_tmp = heap_p->entries[index];
        heap_p->entries[index] = heap_p->entries[index_parent];
        heap_p->entries[index_parent] = entry_tmp;

        bubble_up(heap_p, index_parent);
    }
}

void bubble_down(heap *heap_p, size_t index) {
    if (index == heap_p->num_elems - 1) {
        return;
    }

    size_t index_left = HEAP_LEFT(index);
    size_t index_right = HEAP_RIGHT(index);

    if (index_left > heap_p->num_elems) {
        // No children, so stop here
        return;
    }
    // Only the left child exist, so make both children the same to
    // make all following logic work without the need for exceptions
    index_right = (index_right > heap_p->num_elems) * index_left
                + (index_right <= heap_p->num_elems) * index_right;

    float key_node = heap_p->entries[index].key;
    float key_left = heap_p->entries[index_left].key;
    float key_right = heap_p->entries[index_right].key;

    float key_child;
    size_t index_child;
    if (key_left < key_right) {
        key_child = key_left;
        index_child = index_left;
    } else {
        key_child = key_right;
        index_child = index_right;
    }

    if (key_node > key_child) {
        // Heap order violated, so swap elements
        heap_entry entry_tmp = heap_p->entries[index];
        heap_p->entries[index] = heap_p->entries[index_child];
        heap_p->entries[index_child] = entry_tmp;

        bubble_down(heap_p, index_child);
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
    // printf("%zu after extract\n", heap_p->num_elems);

    return n;
}

void heap_insert(heap *heap_p, size_t n, const float *keys, const void *data) {
    /* Insert n elements into the heap. This function assumes that enough data has
    been allocated to the heap, which can be checked with heap_should_increase_alloc.
    If not, use heap_increase_alloc to double the amount of available memory. */
    // printf("Inserting into heap\n");
    // printf("Current = %zu, alloc = %zu, new = %zu\n", heap_p->num_elems, heap_p->num_alloc, n);

    // for (size_t i = 0; i < heap_p->num_data_arrays; ++ i) {
    //     printf("Data pointer %zu = %p\n", i, heap_p->data[i]);
    // }
    // for (size_t i = 0; i < 2 * heap_p->num_elems; ++ i) {
    //     printf("Data at entry %zu = %p\n", i, heap_p->entries[i].data);
    // }

    for (size_t i = 0; i < n; ++ i) {
        heap_p->entries[heap_p->num_elems].key = keys[i];
        // printf("Got key %f\n", heap_p->entries[heap_p->num_elems].key);
        // printf("Inserting data into %p\n", heap_p->entries[heap_p->num_elems].data);
        memcpy(
            heap_p->entries[heap_p->num_elems].data,
            data + i * heap_p->element_size,
            heap_p->element_size
        );
        // printf("Set data at new entry\n");

        bubble_up(heap_p, heap_p->num_elems);

        ++ heap_p->num_elems;
    }
}

void heap_decrease_key(heap *heap_p, size_t index, float key) {
    heap_p->entries[index].key = key;

    bubble_up(heap_p, index);
}
