#include "values.h"


uint64_t bfs_hash_node(const void *elem, uint64_t seed0, uint64_t seed1) {
    return ((const node *)elem)->hash;
}

int bfs_compare_nodes(const void *elem1, const void *elem2, void *udata) {
    const node *node1 = elem1;
    const node *node2 = elem2;
    return memcmp(node1->state, node2->state, node1->state_size);
}

hashmap *values_node_map_from_states(size_t n, size_t state_size, void *states, action action_space_size) {
    hashmap *map = hashmap_new(sizeof(node), 0, 0, 0, bfs_hash_node, bfs_compare_nodes, NULL, NULL);

    int depth = 0;
    size_t num_states_at_depth = 1;
    size_t next_switch = 1;

    for (size_t i = 0; i < n; ++ i) {
        if (i == next_switch) {
            ++ depth;
            num_states_at_depth *= action_space_size;
            next_switch += num_states_at_depth;
        }
        void *state = states + i * state_size;
        node state_node = {
            .J = depth,
            .state_size = state_size,
            .state = state,
            .hash = hashmap_murmur(state, state_size, 0, 0),
        };

        node *existing_node = hashmap_get(map, &state_node);
        if (existing_node == NULL) {
            hashmap_set(map, &state_node);
        }
    }

    return map;
}

void values_free(hashmap *node_map) {
    hashmap_free(node_map);
}

void values_set(size_t n, size_t state_size, void *states, float *J, hashmap *node_map) {

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        void *state = states + i * state_size;
        node state_node = {
            .J = 1e10,
            .state_size = state_size,
            .state = state,
            .hash = hashmap_murmur(state, state_size, 0, 0),
        };
        node *known_state = hashmap_get(node_map, &state_node);
        if (known_state != NULL) {
            J[i] = known_state->J;
        }
    }
}

