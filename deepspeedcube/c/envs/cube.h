#include <stdlib.h>

#ifndef CUBE_H
#define CUBE_H
typedef char face;
typedef char action;

void multi_act(face *states, action *actions, size_t n);
#endif
