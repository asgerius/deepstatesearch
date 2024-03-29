#ifndef ASTAR_H
#define ASTAR_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "heap.h"

#include "envs/envs.h"
#include "envs/cube.h"
#include "hashmap.c/hashmap.h"

#define NUM_STATES_PER_ARRAY(state_size) (10000000 / (state_size))


struct node {
    action arrival_action;  // The action taken to get to this state
    float g;
    size_t state_size;
    void *state;
    uint64_t hash;
};

struct astar_search {
    float lambda;
    float longest_path;  // Longest path from the initial node to any node
    size_t state_size;
    struct hashmap *node_map;
    heap *frontier;

    size_t num_used_states;
    void *states;
};

typedef struct node node;
typedef struct astar_search astar_search;

astar_search *astar_init(
    float lambda,
    size_t heap_d,
    size_t state_size
);

/* Frees all allocated memory. */
void astar_free(astar_search *search);

void *astar_frontier_ptr(astar_search *search);

void astar_add_initial_state(
    float h,
    void *state,
    astar_search *search
);

/* A* iteration step. A lot of the nomenclature follows the pseudocode on
https://en.wikipedia.org/wiki/A*_search_algorithm. */
void astar_iteration(
    size_t num_current_states,
    const void *current_states,
    size_t num_neighbour_states,
    void *neighbour_states,
    const float *h,
    action *arrival_actions,
    astar_search *search
);

size_t astar_longest_path(astar_search *search);

size_t astar_retrace_path(
    int action_space_size,
    action *actions,  // Actions to solve the initial state go here (in reverse order)
    action *reverse_actions,  // index i gives the reverse of action i
    void *final_state,  // Final state seen that expands to solved state
    void (act)(void *state, action action),
    astar_search *search
);

size_t astar_num_states(astar_search *search_state);

#endif
