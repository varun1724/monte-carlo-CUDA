#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>
#include "utils.h"
#include "simulation_paths.h"

// Forward declaration of the simulation paths kernel.
__global__ void monteCarloPathsKernel(int numSteps, float S0, float r, float sigma,
                                      float dt, float *d_paths, unsigned long seed);

// Declaration of the enhanced regression function.
void runRegressionGPU(const float* h_S, const float* h_Y, int N, float* beta);

// Multi-GPU simulation: generate full price paths.
// The output is stored in h_paths (row-major: each row is one simulation path).
void runMultiGPUSimulationPaths(int numPaths, int numSteps, float S0, float r,
                                float sigma, float T, std::vector<float>& h_paths) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Found " << deviceCount << " CUDA device(s)." << std::endl;
    int pathsPerDevice = (numPaths + deviceCount - 1) / deviceCount;
    h_paths.resize(numPaths * numSteps);
    int offset = 0;
    for (int dev = 0; dev < deviceCount; dev++) {
        CHECK_CUDA(cudaSetDevice(dev));
        int localNumPaths = std::min(pathsPerDevice, numPaths - offset);
        size_t localSize = localNumPaths * numSteps * sizeof(float);
        float *d_paths = nullptr;
        CHECK_CUDA(cudaMalloc(&d_paths, localSize));
        float dt = T / numSteps;
        unsigned long seed = 1234UL + dev;
        int blockSize = 256;
        int gridSize = (localNumPaths + blockSize - 1) / blockSize;
        monteCarloPathsKernel<<<gridSize, blockSize>>>(numSteps, S0, r, sigma, dt, d_paths, seed);
        CHECK_CUDA(cudaDeviceSynchronize());
        // Copy simulation paths from this device to host vector.
        std::vector<float> localPaths(localNumPaths * numSteps);
        CHECK_CUDA(cudaMemcpy(localPaths.data(), d_paths, localSize, cudaMemcpyDeviceToHost));
        for (int i = 0; i < localNumPaths * numSteps; i++) {
            h_paths[(offset * numSteps) + i] = localPaths[i];
        }
        offset += localNumPaths;
        CHECK_CUDA(cudaFree(d_paths));
    }
}

// LSMC Backward Induction for American Option Pricing.
// S_paths: vector containing all simulated asset paths (numPaths x numSteps).
// dt: time step size.
// Returns the estimated American option price.
// Also records intermediate average option prices into convergenceData and corresponding time steps into simulationCounts.
float priceAmericanOptionLSMC(const std::vector<float>& S_paths, int numPaths, int numSteps, float K, float r, float dt,
                                std::vector<float>& convergenceData, std::vector<float>& simulationCounts) {
    // P will hold the option payoff along the paths.
    std::vector<float> P(numPaths * numSteps, 0.0f);
    // At maturity: P(i,T) = max(S(i,T)-K, 0)
    for (int i = 0; i < numPaths; i++) {
        float S_T = S_paths[i * numSteps + (numSteps - 1)];
        P[i * numSteps + (numSteps - 1)] = std::max(S_T - K, 0.0f);
    }
    // Record convergence at maturity.
    {
        double sum = 0.0;
        for (int i = 0; i < numPaths; i++) {
            sum += P[i * numSteps + (numSteps - 1)];
        }
        float avg = static_cast<float>(sum / numPaths);
        convergenceData.push_back(avg);
        simulationCounts.push_back(numSteps - 1);
    }

    // Backward induction.
    for (int t = numSteps - 2; t >= 0; t--) {
        float disc = expf(-r * dt);
        std::vector<float> S_itm;
        std::vector<float> Y_itm;
        for (int i = 0; i < numPaths; i++) {
            float S_val = S_paths[i * numSteps + t];
            float immediate = S_val - K;
            float cont = disc * P[i * numSteps + t + 1];
            if (immediate > 0.0f) {
                S_itm.push_back(S_val);
                Y_itm.push_back(cont);
            } else {
                P[i * numSteps + t] = cont;
            }
        }
        if (!S_itm.empty()) {
            float beta[3] = {0};
            runRegressionGPU(S_itm.data(), Y_itm.data(), S_itm.size(), beta);
            for (int i = 0; i < numPaths; i++) {
                float S_val = S_paths[i * numSteps + t];
                float immediate = S_val - K;
                if (immediate > 0.0f) {
                    float estCont = beta[0] + beta[1] * S_val + beta[2] * (S_val * S_val);
                    // Update payoff: exercise if immediate > continuation.
                    P[i * numSteps + t] = std::max(immediate, disc * P[i * numSteps + t + 1]);
                }
            }
        }
        // Record convergence data (using t as a proxy for time step).
        double sum = 0.0;
        for (int i = 0; i < numPaths; i++) {
            sum += P[i * numSteps + t];
        }
        float avgP = static_cast<float>(sum / numPaths);
        convergenceData.push_back(avgP);
        simulationCounts.push_back(t);
    }
    double sum0 = 0.0;
    for (int i = 0; i < numPaths; i++) {
        sum0 += P[i * numSteps + 0];
    }
    return static_cast<float>(sum0 / numPaths);
}

// Simple CLI visualization: Print an ASCII bar chart of convergence data.
void runCLIViz(const std::vector<float>& x, const std::vector<float>& y) {
    // Find min and max of y.
    float minVal = *std::min_element(y.begin(), y.end());
    float maxVal = *std::max_element(y.begin(), y.end());
    int width = 50;  // Width of ASCII bar.
    std::cout << "\nConvergence Chart (Time step vs. Avg Option Price):\n";
    for (size_t i = 0; i < y.size(); i++) {
        // Scale y to bar length.
        float norm = (y[i] - minVal) / (maxVal - minVal + 1e-6f);
        int barLength = static_cast<int>(norm * width);
        printf("t=%3.0f: ", x[i]);
        for (int j = 0; j < barLength; j++) {
            printf("#");
        }
        printf(" (%.4f)\n", y[i]);
    }
}

// Main function.
int main() {
    // Simulation parameters.
    const int numPaths = 100000;  // Adjust as needed.
    const int numSteps = 100;
    const float S0 = 100.0f;
    const float r = 0.05f;
    const float sigma = 0.2f;
    const float T = 1.0f;
    const float dt = T / numSteps;
    const float K = 100.0f;  // Strike price.

    std::cout << "Starting full LSMC American Option Pricing...\n";

    // Run multi-GPU simulation to obtain full asset price paths.
    std::vector<float> S_paths;
    auto simStart = std::chrono::high_resolution_clock::now();
    runMultiGPUSimulationPaths(numPaths, numSteps, S0, r, sigma, T, S_paths);
    auto simEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> simElapsed = simEnd - simStart;
    std::cout << "Simulation of full paths completed in " << simElapsed.count() * 1000.0 << " ms.\n";

    // Run LSMC backward induction to compute the option price and record convergence data.
    std::vector<float> convergenceData;
    std::vector<float> simulationCounts;
    auto lsmcStart = std::chrono::high_resolution_clock::now();
    float optionPrice = priceAmericanOptionLSMC(S_paths, numPaths, numSteps, K, r, dt, convergenceData, simulationCounts);
    auto lsmcEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> lsmcElapsed = lsmcEnd - lsmcStart;
    std::cout << "LSMC backward induction completed in " << lsmcElapsed.count() * 1000.0 << " ms.\n";
    std::cout << "Estimated American Option Price: " << optionPrice << "\n";

    // For CLI visualization, print an ASCII bar chart of convergence data.
    runCLIViz(simulationCounts, convergenceData);

    // Benchmarking info is printed via timing.
    // Future work: add additional CLI charts for kernel execution times, GPU utilization, etc.

    return 0;
}
