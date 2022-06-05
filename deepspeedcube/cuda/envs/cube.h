#include <stdlib.h>


#ifndef CUBE_H
#define CUBE_H
#define FACES_PER_THREAD 2
typedef char face;
typedef char action;

extern "C" void multi_act(
    char *c_maps_d,
    char *s_maps_d,
    face *states_d,
    action *actions_d,
    size_t n
);
#endif
