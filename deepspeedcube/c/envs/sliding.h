#ifndef SLIDING_H
#define SLIDING_H

#include <stdlib.h>

#include "envs.h"


typedef unsigned char position;

void sliding_multi_act(position *states, action *actions, size_t n);

#endif
