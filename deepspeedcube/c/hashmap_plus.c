#include "hashmap_plus.h"


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
