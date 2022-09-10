#ifndef SLIDING_H
#define SLIDING_H

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "envs.h"

#define ELEMS_PER_STATE(size) ((size) * (size) + 4)
#define ACTION_SPACE 4


/* Actions are 0-3. 0 is move left, 1 is move up, 2 is move right, and 3 is move down. */
void sliding_multi_act(
    short *states,
    const action *actions,
    size_t n,
    short size
);

void sliding_neighbours(
    const short *states,
    const short *neighbours,
    action *actions,
    size_t n,
    short size
);

#endif
