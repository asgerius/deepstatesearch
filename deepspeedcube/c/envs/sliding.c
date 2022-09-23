#include "sliding.h"


sp_dtype state_elems(const sp_dtype *state) {
    sp_dtype size = state[2];
    return size * size + 3;
}

void sliding_act(sp_dtype *state, action action) {
    sp_dtype y = state[0], x = state[1];

    sp_dtype y_new = y + (action - 2) * (action % 2);
    sp_dtype x_new = x - (action - 1) * ((action + 1) % 2);

    state[0] = MAX(MIN(y_new, state[2] - 1), 0);
    state[1] = MAX(MIN(x_new, state[2] - 1), 0);

    sp_dtype index_old = 3 + y * state[2] + x;
    sp_dtype index_new = 3 + state[0] * state[2] + state[1];

    sp_dtype tmp = state[index_old];
    state[index_old] = state[index_new];
    state[index_new] = tmp;
}

void sliding_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
) {
    sp_dtype elems_per_state = state_elems(states);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        action mov = actions[i];
        sp_dtype *state = states + i * elems_per_state;
        sp_dtype y = state[0], x = state[1];

        sp_dtype y_new = y + (mov - 2) * (mov % 2);
        sp_dtype x_new = x - (mov - 1) * ((mov + 1) % 2);

        state[0] = MAX(MIN(y_new, states[2] - 1), 0);
        state[1] = MAX(MIN(x_new, states[2] - 1), 0);

        sp_dtype index_old = 3 + y * state[2] + x;
        sp_dtype index_new = 3 + state[0] * state[2] + state[1];

        sp_dtype tmp = state[index_old];
        state[index_old] = state[index_new];
        state[index_new] = tmp;
    }
}

void sliding_neighbours_set_null_actions(
    const sp_dtype *states,
    const sp_dtype *neighbours,
    action *actions,
    size_t n
) {
    sp_dtype elems_per_state = state_elems(states);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        const sp_dtype *state = states + i * elems_per_state;

        size_t action_index_start = i * ACTION_SPACE;
        const sp_dtype *state_neighbours = neighbours + action_index_start * elems_per_state;

        #pragma unroll
        for (action move = 0; move < ACTION_SPACE; ++ move) {
            const sp_dtype *neighbour = state_neighbours + move * elems_per_state;
            bool changed = memcmp(state, neighbour, 2 * sizeof(sp_dtype));
            size_t action_index = action_index_start + move;
            actions[action_index] = changed * actions[action_index] + !changed * NULL_ACTION;
        }
    }
}
