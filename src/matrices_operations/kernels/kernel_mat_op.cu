#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <cstdlib>
#include "kernel_mat_op.hh"
#include "device_launch_parameters.h"


__global__ void matrixAdditionKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        tmpSum += A[ROW * N + COL] + B[ROW * N + COL];
    }
    C[ROW * N + COL] = tmpSum;
}


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = tmpSum;
}


__global__ void matrixConvolutionKernel(float* A, float* K, float* C, int N, int KN) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;
    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = KN / 2 * -1; i <= KN / 2; i++) {
            for (int j = KN / 2 * -1; j <= KN / 2; j++) {
                if (ROW + i < N && ROW + i >= 0 && COL + j < N && COL + j >= 0)
                    tmpSum += A[(ROW + i) * N + COL + j] * K[(KN / 2 + i) * KN + j + KN / 2];
            }
        }
    }
    C[ROW * N + COL] = tmpSum;
}




void matrixAddition(float *A, float *B, float *C, int N) {
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixAdditionKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);

}

void matrixMultiplication(float *A, float *B, float *C, int N){
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}

void matrixConv(float *A, float *K, float *C, int N, int KN) {
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
    if (N*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixConvolutionKernel<<<blocksPerGrid,threadsPerBlock>>>(A, K, C, N, KN);
}


#include "kernel_mat_op.hh"
