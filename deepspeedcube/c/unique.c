#include "unique.h"


uint64_t hash(const void *elem, uint64_t seed0, uint64_t seed1) {
    const hm_entry *e = elem;
    return hashmap_murmur(e->p_elem, e->bytes, seed0, seed1);
}

int compare(const void *elem1, const void *elem2, void *udata) {
    // Compares two array elements
    const hm_entry *e1 = elem1;
    const hm_entry *e2 = elem2;
    return memcmp(e1->p_elem, e2->p_elem, e1->bytes);
}

hashmap *create_hashmap() {
    return hashmap_new(sizeof(hm_entry*), 0, 0, 0, hash, compare, NULL, NULL);
}

int unique(
    void *array,
    size_t n,
    size_t stride,
    unsigned long *index,
    unsigned long *inverse
) {
    hashmap *map = create_hashmap();

    size_t n_unique = 0;
    size_t i;
    for (i = 0; i < n; ++ i) {
        // Construct element
        const hm_entry this_elem = {
            .p_elem = array + i * stride,
            .bytes = stride,
        };
        // Check if already in map
        const hm_entry *p_found_elem = hashmap_get(map, &this_elem);
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
