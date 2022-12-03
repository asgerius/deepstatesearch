#ifndef CUBE_H
#define CUBE_H

#include <stdio.h>
#include <stdlib.h>

#include "envs.h"


typedef char face;

void cube_print_state(const face *state);

void cube_act(face *state, action action);

void cube_multi_act(face *states, const action *actions, size_t n);

#endif
