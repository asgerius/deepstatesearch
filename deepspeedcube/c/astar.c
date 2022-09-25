#include "astar.h"


uint64_t hash_node(const void *elem, uint64_t seed0, uint64_t seed1) {
    return ((const node *)elem)->hash;
}

int compare_nodes(const void *elem1, const void *elem2, void *udata) {
    const node *node1 = elem1;
    const node *node2 = elem2;
    return memcmp(node1->state, node2->state, node1->state_size);
}

node *init_node(
    action arrival_action,
    float f,
    size_t g,
    size_t state_size,
    void *state,
    uint64_t hash
) {
    node *node_p = malloc(sizeof(node));

    node_p->arrival_action = arrival_action;
    node_p->f = f;
    node_p->g = g;
    node_p->state_size = state_size;
    node_p->state = state;
    node_p->hash = hash;

    return node_p;
}

/* Allocates a new state array for the A* search. The first eight bytes are used as a pointer
to the previous state array, which is replaced in the astar_search struct by the new array.
The first allocated array will point to NULL. */
void astar_new_state_array(astar_search *search, bool is_first) {
    void **states = malloc(
        sizeof(void *) + search->state_size * NUM_STATES_PER_ARRAY(search->state_size)
    );

    states[0] = is_first ? NULL : search->states;

    search->states = states;
    search->num_used_states = 0;
}

astar_search *astar_init(
    float lambda,
    size_t heap_d,
    size_t state_size
) {
    astar_search *search = malloc(sizeof(astar_search));
    search->lambda = lambda;
    search->longest_path = 0;
    search->state_size = state_size;
    search->frontier = heap_alloc(heap_d, state_size);
    search->node_map = hashmap_new(sizeof(node), 0, 0, 0, hash_node, compare_nodes, NULL, NULL);

    astar_new_state_array(search, true);

    return search;
}

void astar_free(astar_search *search) {
    void *next_arr = NULL;
    do {
        next_arr = ((void **)search->states)[0];
        free(search->states);
        search->states = next_arr;
    } while (next_arr != NULL);
    hashmap_free(search->node_map);
    heap_free(search->frontier);
    free(search);
}

void *astar_frontier_ptr(astar_search *search) {
    return search->frontier;
}

void astar_add_initial_state(
    float h,
    void *state,
    astar_search *search
) {
    void *state_p = search->states + sizeof(void *);
    memcpy(state_p, state, search->state_size);
    node *new_node_p = init_node(
        NULL_ACTION, h, 0, search->state_size, state_p,
        hashmap_murmur(state_p, search->state_size, 0, 0)
    );

    search->num_used_states = 1;
    hashmap_set(search->node_map, new_node_p);
    heap_insert(search->frontier, 1, &h, state);

    free(new_node_p);
}

void astar_iteration(
    size_t num_current_states,
    const void *current_states,
    size_t num_neighbour_states,
    void *neighbour_states,
    const float *h,
    action *arrival_actions,
    astar_search *search
) {

    // Get current nodes in parallel
    float *g_current = malloc(num_current_states * sizeof(*g_current));
    uint64_t *hashes = malloc(num_neighbour_states * sizeof(*hashes));
    size_t neighbours_per_state = num_neighbour_states / num_current_states;
    #pragma omp parallel for
    for (size_t i = 0; i < num_current_states; ++ i) {
        node tmp_node = {
            .arrival_action = NULL_ACTION,
            .f = 0, .g = 0,
            .state_size = search->state_size,
            .state = (void *)current_states + i * search->state_size,
            .hash = hashmap_murmur(current_states + i * search->state_size, search->state_size, 0, 0),
        };
        g_current[i] = ((node *)hashmap_get(search->node_map, &tmp_node))->g;

        for (size_t j = 0; j < neighbours_per_state; ++ j) {
            size_t neighbour_index = i * neighbours_per_state + j;
            hashes[neighbour_index] = hashmap_murmur(
                neighbour_states + neighbour_index * search->state_size,
                search->state_size, 0, 0
            );
        }
    }

    // A temporary node used for node lookups. The state pointer is changed
    // such that it is not necessary to create new nodes just for looking up
    // existing nodes. Do NOT use for creating new nodes, as the state pointer
    // may change or be freed. Instead, use init_node, which creates a new node
    // which has its own dedicated memory for the state.
    node tmp_node = {
        .arrival_action = NULL_ACTION,
        .f = 0, .g = 0,
        .state_size = search->state_size,
        .state = NULL,
        .hash = 0,
    };

    // i is the number of the current state, and j is the j'th neighbour of i
    for (size_t i = 0; i < num_current_states; ++ i) {
        size_t g_tentative = g_current[i] + 1;

        if (search->num_used_states + neighbours_per_state > NUM_STATES_PER_ARRAY(search->state_size)) {
            astar_new_state_array(search, false);
        }

        #pragma unroll
        for (action j = 0; j < neighbours_per_state; ++ j) {
            size_t neighbour_index = i * neighbours_per_state + j;

            tmp_node.state = neighbour_states + neighbour_index * search->state_size;
            tmp_node.hash = hashes[neighbour_index];
            node *neighbour_node = hashmap_get(search->node_map, &tmp_node);

            if (neighbour_node != NULL && g_tentative < neighbour_node->g) {
                // Node has been seen before and has shorter path to it
                neighbour_node->f = search->lambda * g_tentative + h[neighbour_index];
                neighbour_node->g = g_tentative;
                neighbour_node->arrival_action = j;
                // if (neighbour_node in search->frontier) {
                //     heap_decrease_key(search->frontier, , neighbour_node->g);
                // }
            } else if (neighbour_node == NULL) {
                // Node has not been seen before, so add to node map and frontier
                void *state_p = search->states + sizeof(void *) + search->num_used_states * search->state_size;
                memcpy(
                    state_p,
                    tmp_node.state,
                    search->state_size
                );
                node *new_node_p = init_node(
                    j,
                    search->lambda * g_tentative + h[neighbour_index],
                    g_tentative,
                    search->state_size,
                    state_p,
                    tmp_node.hash
                );
                hashmap_set(search->node_map, new_node_p);
                heap_insert(search->frontier, 1, &new_node_p->f, new_node_p->state);
                ++ search->num_used_states;
                search->longest_path = MAX(search->longest_path, new_node_p->g);

                free(new_node_p);
            }
        }
    }

    free(g_current);
    free(hashes);
}

size_t astar_longest_path(astar_search *search) {
    return search->longest_path;
}

size_t astar_retrace_path(
    int action_space_size,
    action *actions,  // Actions to solve the initial state go here (in reverse order)
    action *reverse_actions,  // index i gives the reverse of action i
    void *final_state,  // Final state seen that expands to solved state
    void (act)(void *state, action action),
    astar_search *search
) {
    node tmp_node = {
        .f = 0, .g = 0,
        .arrival_action = NULL_ACTION,
        .state_size = search->state_size,
        .state = final_state,
        .hash = hashmap_murmur(final_state, search->state_size, 0, 0),
    };
    node *current_node = hashmap_get(search->node_map, &tmp_node);

    size_t i = 1;
    while (current_node->arrival_action != NULL_ACTION) {
        actions[i] = current_node->arrival_action;
        if (actions[i] == NULL_ACTION) {
            break;
        }

        // At this point, we can fuck up the states, so never mind inplace movements
        action reverse_action = reverse_actions[current_node->arrival_action];
        act(current_node->state, reverse_action);
        current_node->hash = hashmap_murmur(current_node->state, current_node->state_size, 0, 0);
        current_node = hashmap_get(search->node_map, current_node);

        ++ i;
    }

    return i;
}

size_t astar_num_states(astar_search *search_state) {
    return hashmap_count(search_state->node_map);
}
