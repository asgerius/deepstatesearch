#include "sliding.h"


const action all_actions[4] = { 0, 1, 2, 3 };

void sliding_multi_act(
    short *states,
    const action *actions,
    size_t n,
    short size
) {
    size_t elems_per_state = ELEMS_PER_STATE(size);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        action mov = actions[i];
        short *state = states + i * elems_per_state;
        short y = state[0], x = state[1];

        short y_new = y - (mov - 1) * ((mov + 1) % 2);
        short x_new = x + (mov - 2) * (mov % 2);

        state[0] = MAX(MIN(y_new, size-1), 0);
        state[1] = MAX(MIN(x_new, size-1), 0);

        short index_old = 2 + x * size + x;
        short index_new = 2 + state[1] * size + state[0];

        short tmp = state[index_old];
        state[index_old] = state[index_new];
        state[index_new] = tmp;
    }
}

void sliding_neighbours_set_null_actions(
    const short *states,
    const short *neighbours,
    action *actions,
    size_t n,
    short size
) {
    size_t elems_per_state = ELEMS_PER_STATE(size);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        const short *state = states + i * elems_per_state;

        size_t action_index_start = i * ACTION_SPACE * elems_per_state;
        const short *state_neighbours = neighbours + i * ACTION_SPACE * elems_per_state;

        #pragma unroll
        for (action move = 0; move < ACTION_SPACE; ++ i) {
            const short *neighbour = state_neighbours + move * elems_per_state;
            bool changed = memcmp(state, neighbour, 2 * sizeof(*state));
            size_t action_index = action_index_start + move;
            actions[action_index] = changed * actions[action_index] + !changed * NULL_ACTION;
        }
    }
}
