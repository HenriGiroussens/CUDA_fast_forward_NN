#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <cstdlib>
#include "kernel_mat_op.hh"
#include "device_launch_parameters.h"


__global__ void matrixAdditionKernel(float* A, float* B, float* C, int N, int M) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;


    if (ROW < N && COL < M) {
        // each thread computes one element of the block sub-matrix
        tmpSum += A[ROW * M + COL] + B[ROW * M + COL];
    }
    C[ROW * M + COL] = tmpSum;
}


__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int NA, int MA, int NB, int MB) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;

    if (ROW < NA && COL < MB) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < MA; i++) {
            tmpSum += A[ROW * MA + i] * B[i * MB + COL];
        }
    }
    C[ROW * MB + COL] = tmpSum;
}


__global__ void matrixConvolutionKernel(float* A, float* K, float* C, int N, int M, int KN) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float tmpSum = 0;
    if (ROW < N && COL < M) {
        // each thread computes one element of the block sub-matrix
        for (int i = KN / 2 * -1; i <= KN / 2; i++) {
            for (int j = KN / 2 * -1; j <= KN / 2; j++) {
                if (ROW + i < N && ROW + i >= 0 && COL + j < M && COL + j >= 0)
                    tmpSum += A[(ROW + i) * M + COL + j] * K[(KN / 2 + i) * KN + j + KN / 2];
            }
        }
    }
    C[ROW * M + COL] = tmpSum;
}




void matrixAddition(float *A, float *B, float *C, int N, int M) {
    dim3 threadsPerBlock(M, N);
    dim3 blocksPerGrid(1, 1);
    if (N*M > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(M)/double(threadsPerBlock.y));
    }
    matrixAdditionKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N, M);

}

void matrixMultiplication(float *A, float *B, float *C, int NA, int MA, int NB, int MB){
    dim3 threadsPerBlock(MB, NA);
    dim3 blocksPerGrid(1, 1);
    if (NA*NA > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(NA)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(NA)/double(threadsPerBlock.y));
    }
    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, NA, MA, NB, MB);
}

void matrixConv(float *A, float *K, float *C, int N, int M, int KN) {
    dim3 threadsPerBlock(M, N);
    dim3 blocksPerGrid(1, 1);
    if (M*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(M)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixConvolutionKernel<<<blocksPerGrid,threadsPerBlock>>>(A, K, C, N, M, KN);
}


#include "kernel_mat_op.hh"
