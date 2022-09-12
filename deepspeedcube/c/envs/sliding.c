#include "sliding.h"


const action all_actions[4] = { 0, 1, 2, 3 };

void sliding_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n,
    sp_dtype size
) {
    size_t elems_per_state = ELEMS_PER_STATE(size);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        action mov = actions[i];
        sp_dtype *state = states + i * elems_per_state;
        sp_dtype y = state[0], x = state[1];

        sp_dtype y_new = y + (mov - 2) * (mov % 2);
        sp_dtype x_new = x - (mov - 1) * ((mov + 1) % 2);

        state[0] = MAX(MIN(y_new, size-1), 0);
        state[1] = MAX(MIN(x_new, size-1), 0);

        sp_dtype index_old = 2 + y * size + x;
        sp_dtype index_new = 2 + state[0] * size + state[1];

        sp_dtype tmp = state[index_old];
        state[index_old] = state[index_new];
        state[index_new] = tmp;
    }
}

void sliding15_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
) {
    sliding_multi_act(states, actions, n, 4);
}

void sliding24_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
) {
    sliding_multi_act(states, actions, n, 5);
}

void sliding35_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
) {
    sliding_multi_act(states, actions, n, 6);
}

void sliding48_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
) {
    sliding_multi_act(states, actions, n, 7);
}

void sliding63_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
) {
    sliding_multi_act(states, actions, n, 8);
}

void sliding_neighbours_set_null_actions(
    const sp_dtype *states,
    const sp_dtype *neighbours,
    action *actions,
    size_t n,
    sp_dtype size
) {
    size_t elems_per_state = ELEMS_PER_STATE(size);

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
