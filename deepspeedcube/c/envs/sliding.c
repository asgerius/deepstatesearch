#include "sliding.h"

void sliding_multi_act(
    short int *states,
    action *actions,
    size_t n,
    size_t width,
    size_t height
) {
    size_t state_size = STATE_SIZE(width, height);
    
    #pragma omp parallel for
    for (size_t i = 0; i < n; ++ i) {
        action mov = actions[i];
        short int *state = states + i * state_size;
        short int x = state[0], y = state[1];

        short int x_new = x - (mov - 1) * ((mov + 1) % 2);
        short int y_new = y - (mov - 2) * (mov % 2);

        x_new = MAX(MIN(x_new, 127), 0);
        y_new = MAX(MIN(y_new, 127), 0);

        short int index_old = 2 + y * width + x;
        short int index_new = 2 + y_new * width + x_new;

        short int tmp = state[index_old];
        state[index_old] = state[index_new];
        state[index_new] = tmp;
    }
}
