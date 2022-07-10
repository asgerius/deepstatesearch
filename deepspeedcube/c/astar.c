#include "astar.h"


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
