# Monte Carlo American Option Pricing with CUDA

This project implements a GPU-accelerated Monte Carlo simulation for pricing American options using the Least-Squares Monte Carlo (LSMC) method. The project is written in C++ with CUDA and uses cuBLAS for matrix operations (for the regression step). In later stages, real-time visualization (using third-party libraries like Dear ImGui/ImPlot) and multi-GPU benchmarking will be added.

## Features

- **CUDA-accelerated Path Simulation:**  
  Each GPU thread simulates a price path using geometric Brownian motion.
  
- **Least-Squares Monte Carlo (LSMC):**  
  (Planned) Backward induction using regression to decide early exercise.

- **cuBLAS Integration:**  
  Use cuBLAS for efficient matrix computations in the regression step.

- **Visualization:**  
  (Planned) Real-time display of simulation convergence and performance metrics.

- **Multi-GPU Benchmarking:**  
  (Planned) Scale the simulation across multiple GPUs.

## File Structure

MonteCarloAmerican/
├── README.md           # This file: project description and instructions.
└── src/
    ├── main.cpp        # Main application: sets up simulation, calls kernels.
    ├── simulation.cu   # CUDA kernels for Monte Carlo path simulation.
    ├── regression.cu   # (Stub) for regression routines (LSMC).
    └── utils.h         # Utility macros and helper functions.

## Build Instructions

From the project root, you can use CMake:

```bash
mkdir build && cd build
cmake ..
make

```

Alternatively, compile manually with:

```bash
nvcc -O3 -arch=sm_75 ../src/main.cpp ../src/simulation.cu ../src/regression.cu -lcublas -o MonteCarloAmerican
```

## Running the Project
Simply run:

```bash
./MonteCarloAmerican
```

The program will run the Monte Carlo simulation kernel (with a basic European simulation as a starting point), print benchmark timing results, and (eventually) display visualization windows.

## Future Enhancements
- Implement regression for the LSMC method.

- Add real-time visualization using Dear ImGui/ImPlot.

- Enable multi-GPU simulation and benchmarking.