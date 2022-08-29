#ifndef SLIDING_H
#define SLIDING_H

#include <stdlib.h>

#include "envs.h"

#define STATE_SIZE(width, height) ((width) * (height) + 2)


/* Actions are 0-3. 0 is move left, 1 is move up, 2 is move right, and 3 is move down. */
void sliding_multi_act(
    short int *states,
    action *actions,
    size_t n,
    size_t width,
    size_t height
);

#endif
