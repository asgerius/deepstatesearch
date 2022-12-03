#include <stdlib.h>
#include <cuda.h>
#include "cube.h"

#define K 16


__global__ void _multi_act(
    char *maps_d,
    face *states_d,
    action *actions_d,
    size_t n
) {
    size_t tidy = threadIdx.x + blockIdx.x * blockDim.x;

    if (tidy < n) {
        // Performs n actions_d on n states in-place
        int tidx = threadIdx.y + blockIdx.y * blockDim.y;
        int idx = FACES_PER_THREAD * tidx;
        char *ptr = maps_d + 12 * 24 * (idx >= 8) + 24 * actions_d[tidy];
        face *face_ptr = states_d + 20 * tidy + idx;
        int i;
        #pragma unroll
        for (i = 0; i < FACES_PER_THREAD; ++ i) {
            face_ptr[i] = ptr[face_ptr[i]];
        }
    }
}

extern "C" void multi_act(
    char *maps_d,
    face *states_d,
    action *actions_d,
    size_t n
) {
    _multi_act<<<(n-1) / K + 1, dim3(K, 20/FACES_PER_THREAD)>>>(
        maps_d, states_d, actions_d, n
    );
    cudaDeviceSynchronize();
}
