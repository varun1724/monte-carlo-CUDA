#include <cuda_runtime.h>
#include "utils.h"
#include <curand_kernel.h>

extern "C" __global__ void monteCarloPathsKernel(int numSteps, float S0, float r, float sigma,
                                                  float dt, float *d_paths, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState_t state;
    curand_init(seed, idx, 0, &state);
    float S = S0;
    d_paths[idx * numSteps] = S;
    for (int i = 1; i < numSteps; i++) {
        float Z = curand_normal(&state);
        S *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * Z);
        d_paths[idx * numSteps + i] = S;
    }
}
