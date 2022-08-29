#ifndef ASTAR_H
#define ASTAR_H

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "heap.h"

#include "envs/envs.h"
#include "envs/cube.h"
#include "hashmap.c/hashmap.h"

#define NULL_ACTION (action)UCHAR_MAX


struct node {
    action arrival_action;  // The action taken to get to this state
    float f;
    size_t g;
    size_t state_size;
    void *state;
};

struct astar_search {
    float lambda;
    size_t longest_path;  // Longest path from the initial node to any node
    size_t state_size;
    struct hashmap *node_map;
    heap *frontier;
};

typedef struct node node;
typedef struct astar_search astar_search;

astar_search *astar_init(
    float lambda,
    size_t state_size,
    heap *frontier
);

/* Frees all allocated memory and returns number of states in the node map. */
size_t astar_free(astar_search *search);

void astar_add_initial_state(
    float h,
    void *state,
    astar_search *search
);

/* A* iteration step. A lot of the nomenclature follows the pseudocode on
https://en.wikipedia.org/wiki/A*_search_algorithm. It returns the size of
the frontier (including the null element). */
size_t astar_insert_neighbours(
    size_t num_current_states,  // 1 until batched A* is implemented
    void *current_states,
    size_t num_neighbour_states,
    void *neighbour_states,
    float *h,
    action *arrival_actions,
    astar_search *search
);

size_t astar_longest_path(astar_search *search);

size_t astar_retrace_path(
    int action_space_size,
    action *actions,  // Actions to solve the initial state go here (in reverse order)
    action *reverse_actions,  // index i gives the reverse of action i
    void *final_state,  // Final state seen that expands to solved state
    void (act)(void *state, void *action, size_t num_actions),
    astar_search *search
);

#endif
