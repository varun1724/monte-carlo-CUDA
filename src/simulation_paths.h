#pragma once

#ifdef __cplusplus
extern "C" {
#endif

__global__ void monteCarloPathsKernel(int numSteps, float S0, float r, float sigma,
                                       float dt, float *d_paths, unsigned long seed);

#ifdef __cplusplus
}
#endif
