#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <cstddef>
#include <cassert>
#include <cuda_runtime.h>
#include "kernel_mat_op.hh"



int mat_add(float** A, float** B, int N)
{
    cudaError_t rc = cudaSuccess;

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int SIZE = N*N;

    // Allocate memory on the host
    std::vector<float> h_A(SIZE);
    std::vector<float> h_B(SIZE);
    std::vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = A[i][j];
            h_B[i*N+j] = B[i][j];
        }
    }

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;

    rc = cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_C, SIZE * sizeof(float));
    if (rc)
        return -1;

    // Copy to device
    cudaMemcpy(&d_A, &h_A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_B, &h_B, SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(&d_C, 0, SIZE * sizeof(float));

    matrixAddition(d_A, d_B, d_C, N);


    return 0;
}