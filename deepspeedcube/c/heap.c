#include "heap.h"


void bubble_up(heap heap, size_t index) {
    if (index == 1) {
        return;
    }

    size_t index_parent = HEAP_PARENT(index);

    float key_node = heap.keys[index];
    float key_parent = heap.keys[index_parent];

    if (key_node < key_parent) {
        // Heap order violated
        heap.keys[index] = key_parent;
        heap.keys[index_parent] = key_node;

        void *data_tmp = malloc(heap.element_size);
        void *data_node = heap.data + index * heap.element_size;
        void *data_parent = heap.data + index_parent * heap.element_size;

        memcpy(data_tmp,    data_node,   heap.element_size);
        memcpy(data_node,   data_parent, heap.element_size);
        memcpy(data_parent, data_tmp,    heap.element_size);

        bubble_up(heap, index_parent);
    } else {
        return;
    }
}

void bubble_down(heap heap, size_t index);


heap_entry heap_extract_min(heap heap) {
    float key_min = heap.keys[1];
    void *data_min = malloc(heap.element_size);

    void *first_data_element = heap.data + heap.element_size;
    memcpy(
        data_min,
        first_data_element,
        heap.element_size
    );
    memcpy(
        first_data_element,
        heap.data + heap.element_size * heap.num_elems,
        heap.element_size
    );

    -- heap.num_elems;

    bubble_down(heap, 1);

    heap_entry entry_min = {
        .key = key_min,
        .data = data_min,
    };
    return entry_min;
}

void heap_insert(heap heap, heap_entry entry) {
    ++ heap.num_elems;

    heap.keys[heap.num_elems] = entry.key;
    memcpy(
        heap.data + heap.num_elems * heap.element_size,
        entry.data,
        heap.element_size
    );

    bubble_up(heap, heap.num_elems);
}

void heap_decrease_key(heap heap, size_t index, float key) {
    heap.keys[index] = key;

    bubble_up(heap, index);
}
