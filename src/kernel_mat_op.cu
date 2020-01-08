#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <cstdlib>
#include "kernel_mat_op.hh"

__global__ void matrixAdditionKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        tmpSum += A[ROW * N + COL] + B[ROW * N + COL];
    }
    C[ROW * N + COL] = tmpSum;
}

void matrixAddition(float *A, float *B, float *C, int N) {
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    matrixAdditionKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);

}

#include "kernel_mat_op.hh"
