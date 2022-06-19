#include "astar.h"


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

hashmap *new_hashmap() {
    return hashmap_new(sizeof(hm_entry*), 0, 0, 0, hash, compare, NULL, NULL);
}

void already_exist(
    size_t n,
    size_t stride,
    bool *new,
    void *new_states,
    hashmap *seen_states
) {
    size_t i;
    for (i = 0; i < n; ++ i) {
        const hm_entry this_elem = {
            .p_elem = new_states + stride * i,
            .bytes = stride,
        };
        const hm_entry *p_found_entry = hashmap_get(seen_states, &this_elem);

        if (p_found_entry != NULL) {
            new[i] = false;
        } else {
            new[i] = true;
            hashmap_set(seen_states, &this_elem);
        }
    }
}
