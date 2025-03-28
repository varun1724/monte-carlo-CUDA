// regression.cu
#include "utils.h"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Enhanced GPU regression using cuBLAS.
// Given host arrays h_S and h_Y (each of length N) representing the independent variable (S)
// and the dependent variable (Y, e.g., discounted payoffs), this function builds the design matrix X
// where each row is [1, S, S^2], computes X^T * X (a 3x3 matrix) and X^T * Y (a 3x1 vector) on the GPU,
// and then solves the system (X^T * X) * beta = X^T * Y on the host using Gaussian elimination.
// The solution beta (array of length 3) is returned via the beta pointer.
void runRegressionGPU(const float* h_S, const float* h_Y, int N, float* beta) {
    // Build design matrix X on the host: dimensions N x 3.
    int numRows = N;
    int numCols = 3;
    size_t size_X = numRows * numCols * sizeof(float);
    float* h_X = (float*)malloc(size_X);
    for (int i = 0; i < N; i++) {
        h_X[i * 3 + 0] = 1.0f;
        h_X[i * 3 + 1] = h_S[i];
        h_X[i * 3 + 2] = h_S[i] * h_S[i];
    }

    // Allocate device memory for X and Y.
    float *d_X = nullptr, *d_Y = nullptr;
    CHECK_CUDA(cudaMalloc(&d_X, size_X));
    CHECK_CUDA(cudaMalloc(&d_Y, numRows * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_X, h_X, size_X, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Y, h_Y, numRows * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle.
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Compute X^T * X: (3 x N) * (N x 3) = 3 x 3 matrix.
    float A[9] = {0}; // Host copy of X^T * X.
    float *d_A = nullptr;
    size_t size_A = 3 * 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_A, size_A));
    float alpha_cublas = 1.0f, beta_cublas = 0.0f;
    // X is numRows x 3. We want A = (X^T) * X.
    // In cuBLAS, we set op(A)=Transpose, op(B)=NoTranspose.
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             3, 3, numRows,
                             &alpha_cublas,
                             d_X, numRows,  // X in column-major is transposed relative to row-major.
                             d_X, numRows,
                             &beta_cublas,
                             d_A, 3));
    CHECK_CUDA(cudaMemcpy(A, d_A, size_A, cudaMemcpyDeviceToHost));

    // Compute X^T * Y: (3 x N) * (N x 1) = 3 x 1 vector.
    float b[3] = {0};
    float *d_b = nullptr;
    size_t size_b = 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_b, size_b));
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             3, 1, numRows,
                             &alpha_cublas,
                             d_X, numRows,
                             d_Y, numRows,
                             &beta_cublas,
                             d_b, 3));
    CHECK_CUDA(cudaMemcpy(b, d_b, size_b, cudaMemcpyDeviceToHost));

    // Solve the 3x3 system A * beta = b on the host using Gaussian elimination.
    const int n = 3;
    float aug[n][n+1];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i][j] = A[i*n + j];
        }
        aug[i][n] = b[i];
    }
    for (int i = 0; i < n; i++) {
        float pivot = aug[i][i];
        if (fabs(pivot) < 1e-6f) {
            fprintf(stderr, "Pivot too small in regression solve.\n");
            exit(EXIT_FAILURE);
        }
        for (int j = i; j < n+1; j++) {
            aug[i][j] /= pivot;
        }
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = aug[k][i];
                for (int j = i; j < n+1; j++) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }
    for (int i = 0; i < n; i++) {
        beta[i] = aug[i][n];
    }
    printf("GPU Regression coefficients: %f, %f, %f\n", beta[0], beta[1], beta[2]);

    // Clean up.
    free(h_X);
    CHECK_CUDA(cudaFree(d_X));
    CHECK_CUDA(cudaFree(d_Y));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUBLAS(cublasDestroy(handle));
}
