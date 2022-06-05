#include <stdlib.h>
#include <cuda.h>
#include "cube.h"
#include <stdio.h>

#define K 16


__global__ void _multi_act(
    char *c_maps_d,
    char *s_maps_d,
    face *states_d,
    action *actions_d,
    size_t n
) {
    int tidy = threadIdx.x + blockIdx.x * blockDim.x;
    int tidx = threadIdx.y + blockIdx.y * blockDim.y;

    if (tidy < n) {
        // Performs n actions_d on n states in-place
        // Row pointer for easy indexing
        face *p_state = states_d + 20 * tidy;
        // Slightly faster by only looking up maps once
        const action action = actions_d[tidy];
        // Map corners
        int i;
        if (FACES_PER_THREAD * tidx < 8) {
            #pragma unroll
            for (i = 0; i < FACES_PER_THREAD; i ++) {
                face *face_ptr = p_state + FACES_PER_THREAD * tidx + i;
                *face_ptr = (c_maps_d+24*action)[*face_ptr];
            }
        } else {
            #pragma unroll
            for (i = 0; i < FACES_PER_THREAD; i ++) {
                face *face_ptr = p_state + FACES_PER_THREAD * tidx + i;
                *face_ptr = (s_maps_d+24*action)[*face_ptr];
            }
        }
    }
}

extern "C" void multi_act(
    char *c_maps_d,
    char *s_maps_d,
    face *states_d,
    action *actions_d,
    size_t n
) {
    _multi_act<<<dim3((n-1)/(20*K)+1, 1), dim3(K, 20/FACES_PER_THREAD)>>>(
        c_maps_d, s_maps_d, states_d, actions_d, n
    );
    cudaDeviceSynchronize();
}
