#include "kernel_mat_op.hh"



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


__device__ float relu(float x) {
    return x > 0 ? x : 0;
}

__device__ float sigmoid(float x) {
    return 1 / (1 + std::exp(-x));
}


__global__ void matrixApplyFunctionKernel(float* A, float* B, int N, int func_id) {
    int POS = blockIdx.x * blockDim.x + threadIdx.x;
    if (POS < N) {
        if (func_id == 1) {
            B[POS] = relu(A[POS]);
        }
        else if (func_id == 2) {
            B[POS] = std::tanh(A[POS]);
        }
        else if (func_id == 3) {
            B[POS] = sigmoid(A[POS]);
        }
        else if (func_id == 4) {
            B[POS] = std::exp(A[POS]);
        } else {
            B[POS] = A[POS];
        }
    }
}

__global__ void matrixApplySoftmaxKernel(float* A, float* B, int N, float sum) {
    int POS = blockIdx.x*blockDim.x+threadIdx.x;
    if (POS < N) {
        B[POS] = A[POS] / sum;
    }
}


__global__ void matrixApplySumKernel(float* A, float* res, int N) {
    int idx = threadIdx.x;
    const int blockSize = 1024;
    int sum = 0;
    for (int i = idx; i < N; i += blockSize)
        sum += A[i];
    __shared__ int r[blockSize];
    r[idx] = sum;
    __syncthreads();
    for (int size = blockSize/2; size>0; size/=2) {
        if (idx<size)
            r[idx] += r[idx+size];
        __syncthreads();
    }
    if (idx == 0)
        *res = r[0];
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

void matrixApplyFunction(float* A, float* B, int N, std::string func) {
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
    if (N > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }
    int func_id = -1;
    if (func == "relu")
        func_id = 1;
    if (func == "tanh")
        func_id = 2;
    if (func == "sigmoid")
        func_id = 3;
    if (func == "exp")
        func_id = 4;
    matrixApplyFunctionKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, N, func_id);
}


void matrixApplySoftmax(float* A, float* B, int N, float sum) {
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
    if (N > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }
    matrixApplySoftmaxKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, N, sum);
}


float matrixSum(float* A, int N) {
    dim3 threadsPerBlock(1024);
    dim3 blocksPerGrid(1);
    if (N > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }
    float res = 0;
    matrixApplySumKernel<<<blocksPerGrid,threadsPerBlock>>>(A, &res, N);
    return res;
}


#include <cmath>

#include "kernel_mat_op.hh"
