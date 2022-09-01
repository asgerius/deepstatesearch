#include "sliding.h"

void sliding_multi_act(
    short *states,
    action *actions,
    size_t n,
    short size
) {
    size_t state_size = STATE_SIZE(size);

    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        action mov = actions[i];
        short *state = states + i * state_size;
        short x = state[0], y = state[1];

        short x_new = x - (mov - 1) * ((mov + 1) % 2);
        short y_new = y + (mov - 2) * (mov % 2);

        state[0] = MAX(MIN(x_new, size-1), 0);
        state[1] = MAX(MIN(y_new, size-1), 0);

        short index_old = 2 + y * size + x;
        short index_new = 2 + state[1] * size + state[0];

        short tmp = state[index_old];
        state[index_old] = state[index_new];
        state[index_new] = tmp;
    }
}
