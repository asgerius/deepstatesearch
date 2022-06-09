#include "unique.h"


// Contains a pointer to an array element and a reference to the stride
struct elem {
    void *p_elem;
    size_t stride;
};

typedef struct hashmap hashmap;

uint64_t hash(const void *elem, uint64_t seed0, uint64_t seed1) {
    const struct elem *e = elem;
    return hashmap_murmur(e->p_elem, e->stride, seed0, seed1);
}

int compare(const void *elem1, const void *elem2, void *udata) {
    // Compares two array elements
    const struct elem *e1 = elem1;
    const struct elem *e2 = elem2;
    return memcmp(e1->p_elem, e2->p_elem, e1->stride);
}

int unique(
    void *array,
    size_t n,
    size_t stride,
    unsigned long *index,
    unsigned long *inverse
) {
    hashmap *map = hashmap_new(sizeof(struct elem*), 0, 0, 0, hash, compare, NULL, NULL);

    size_t n_unique = 0;
    size_t i;
    for (i = 0; i < n; i ++) {
        // Construct element
        struct elem this_elem = {
            .p_elem = array + stride * i,
            .stride = stride,
        };
        // Check if already in map
        struct elem *p_found_elem = hashmap_get(map, &this_elem);
        if (p_found_elem != NULL) {
            // Get index of found element by difference in memory address
            size_t found_index = (p_found_elem->p_elem - array) / stride;
            inverse[i] = inverse[found_index];
        } else {
            // Set new element in hashmap
            hashmap_set(map, &this_elem);
            index[n_unique] = i;
            inverse[i] = n_unique;
            n_unique ++;
        }
    }
    hashmap_free(map);

    return n_unique;
}