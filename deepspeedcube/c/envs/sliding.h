#ifndef SLIDING_H
#define SLIDING_H

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "envs.h"

#define ELEMS_PER_STATE(size) ((size) * (size) + 2)
#define ACTION_SPACE 4


/* Actions are 0-3. 0 is move right, 1 is move up, 2 is move left, and 3 is move down. */
void sliding_multi_act(
    short *states,
    const action *actions,
    size_t n,
    short size
);

void sliding15_multi_act(
    short *states,
    const action *actions,
    size_t n
);

void sliding24_multi_act(
    short *states,
    const action *actions,
    size_t n
);

void sliding35_multi_act(
    short *states,
    const action *actions,
    size_t n
);

void sliding48_multi_act(
    short *states,
    const action *actions,
    size_t n
);

void sliding63_multi_act(
    short *states,
    const action *actions,
    size_t n
);

void sliding_neighbours(
    const short *states,
    const short *neighbours,
    action *actions,
    size_t n,
    short size
);

#endif
