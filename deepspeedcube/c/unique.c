#include "unique.h"


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
