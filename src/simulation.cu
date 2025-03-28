#include <cuda_runtime.h>
#include "utils.h"
#include <curand_kernel.h>

// Kernel: Each thread simulates one path.
// For simplicity, this kernel only computes the final asset price.
// In a full LSMC implementation, you might store intermediate steps.
__global__ void monteCarloKernel(int numSteps, float S0, float r, float sigma, float dt, float *d_results, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize cuRAND state
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    float S = S0;
    for (int i = 0; i < numSteps; i++) {
        float Z = curand_normal(&state);  // standard normal random number
        S *= expf((r - 0.5f * sigma * sigma) * dt + sigma * sqrtf(dt) * Z);
    }
    
    // Store final price of the simulated path
    if (d_results != nullptr)
        d_results[idx] = S;
}

// Host function to launch the simulation kernel.
void runSimulation(int numPaths, int numSteps, float S0, float r, float sigma, float T, float* d_results) {
    float dt = T / numSteps;
    // If d_results is null, allocate temporary device memory.
    float *d_temp = d_results;
    if (d_temp == nullptr) {
        CHECK_CUDA(cudaMalloc(&d_temp, numPaths * sizeof(float)));
    }

    int blockSize = 256;
    int gridSize = (numPaths + blockSize - 1) / blockSize;
    unsigned long seed = 1234UL; // For reproducibility

    monteCarloKernel<<<gridSize, blockSize>>>(numSteps, S0, r, sigma, dt, d_temp, seed);
    CHECK_CUDA(cudaDeviceSynchronize());

    // If we allocated memory here, caller should free it later.
    if (d_results == nullptr) {
        cudaFree(d_temp);
    }
}
