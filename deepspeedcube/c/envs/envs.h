#ifndef ENVS_H
#define ENVS_H

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define MIN(x, y) (((x) < (y)) * (x) + ((x) >= (y)) * (y))
#define MAX(x, y) (((x) >= (y)) * (x) + ((x) < (y)) * (y))

typedef char action;

#define NULL_ACTION (action)UCHAR_MAX

void *copy_state(void *state, size_t state_size);

#endif
