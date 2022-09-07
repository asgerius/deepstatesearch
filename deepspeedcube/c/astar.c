#include "astar.h"


uint64_t hash_node(const void *elem, uint64_t seed0, uint64_t seed1) {
    const node *node = elem;
    return hashmap_murmur(node->state, node->state_size, seed0, seed1);
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
    void *state
) {
    node *node_p = malloc(sizeof(node));

    node_p->arrival_action = arrival_action;
    node_p->f = f;
    node_p->g = g;
    node_p->state_size = state_size;
    node_p->state = malloc(state_size);
    memcpy(node_p->state, state, state_size);

    return node_p;
}

void free_node(void *elem) {
    node *node = elem;
    free(node->state);
}

astar_search *astar_init(
    float lambda,
    size_t state_size
) {
    astar_search *search = malloc(sizeof(astar_search));
    search->lambda = lambda;
    search->longest_path = 0;
    search->state_size = state_size;
    search->frontier = heap_alloc(state_size);
    search->node_map = hashmap_new(sizeof(node), 0, 0, 0, hash_node, compare_nodes, free_node, NULL);

    return search;
}

size_t astar_free(astar_search *search) {
    size_t num_states = hashmap_count(search->node_map);
    hashmap_free(search->node_map);
    heap_free(search->frontier);
    free(search);
    return num_states;
}

void *astar_frontier_ptr(astar_search *search) {
    return search->frontier;
}

void astar_add_initial_state(
    float h,
    void *state,
    astar_search *search
) {
    node *new_node_p = init_node(
        NULL_ACTION, h, 0, search->state_size, state
    );

    hashmap_set(search->node_map, new_node_p);
    heap_insert(search->frontier, 1, &h, state);
}

void astar_insert_neighbours(
    size_t num_current_states,
    void *current_states,
    size_t num_neighbour_states,
    void *neighbour_states,
    float *h,
    action *arrival_actions,
    astar_search *search
) {

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
    };

    // i is the number of the current state, and j is the j'th neighbour of i
    size_t neighbours_per_state = num_neighbour_states / num_current_states;
    for (size_t i = 0; i < num_current_states; ++ i) {
        void *current = current_states + i * search->state_size;

        tmp_node.state = current;
        node *current_node = hashmap_get(search->node_map, &tmp_node);
        size_t g_current = current_node->g;

        #pragma unroll
        for (action j = 0; j < neighbours_per_state; ++ j) {
            size_t neighbour_index = i * neighbours_per_state + j;
            void *neighbour = neighbour_states + neighbour_index * search->state_size;

            size_t g_tentative = g_current + 1;
            tmp_node.state = neighbour;
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
                node *new_node_p = init_node(
                    j,
                    search->lambda * g_tentative + h[neighbour_index],
                    g_tentative,
                    search->state_size,
                    neighbour
                );
                hashmap_set(search->node_map, new_node_p);
                search->longest_path = MAX(search->longest_path, new_node_p->g);
                heap_insert(search->frontier, 1, &new_node_p->f, new_node_p->state);
            }
        }
    }
}

size_t astar_longest_path(astar_search *search) {
    return search->longest_path;
}

size_t astar_retrace_path(
    int action_space_size,
    action *actions,  // Actions to solve the initial state go here (in reverse order)
    action *reverse_actions,  // index i gives the reverse of action i
    void *final_state,  // Final state seen that expands to solved state
    void (act)(void *state, void *action, size_t num_actions),
    astar_search *search
) {
    node tmp_node = {
        .f = 0, .g = 0,
        .arrival_action = NULL_ACTION,
        .state_size = search->state_size,
        .state = final_state,
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
        act(current_node->state, &reverse_action, 1);
        current_node = hashmap_get(search->node_map, current_node);

        ++ i;
    }

    return i;
}
