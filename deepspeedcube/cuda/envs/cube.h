#include <stdlib.h>

#ifndef CUBE_H
#define CUBE_H
#define FACES_PER_THREAD 4

typedef char face;
typedef char action;

extern "C" void multi_act(
    char *maps_d,
    face *states_d,
    action *actions_d,
    size_t n
);

#endif
