#ifndef SLIDING_H
#define SLIDING_H

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "envs.h"

#define ELEMS_PER_STATE(size) ((size) * (size) + 2)
#define ACTION_SPACE 4


typedef signed char sp_dtype;

/* Actions are 0-3. 0 is move right, 1 is move up, 2 is move left, and 3 is move down. */
const action all_actions[4] = { 0, 1, 2, 3 };

void sliding_act(sp_dtype *state, action action);

void sliding_multi_act(
    sp_dtype *states,
    const action *actions,
    size_t n
);

void sliding_neighbours(
    const sp_dtype *states,
    const sp_dtype *neighbours,
    action *actions,
    size_t n
);

#endif
