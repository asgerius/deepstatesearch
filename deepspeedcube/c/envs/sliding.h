#ifndef SLIDING_H
#define SLIDING_H

#include <stdlib.h>

#include "envs.h"

#define STATE_SIZE(size) (2 * (size) * (size) + 4)


/* Actions are 0-3. 0 is move left, 1 is move up, 2 is move right, and 3 is move down. */
void sliding_multi_act(
    short *states,
    action *actions,
    size_t n,
    short size
);

#endif
