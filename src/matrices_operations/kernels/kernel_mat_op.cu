#include "kernel_mat_op.hh"



__global__ void matrixAdditionKernel(double* A, double* B, double* C, int N, int M) {

    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    double tmpSum = 0;


    if (ROW < N && COL < M) {
        // each thread computes one element of the block sub-matrix
        tmpSum += A[ROW * M + COL] + B[ROW * M + COL];
    }
    C[ROW * M + COL] = tmpSum;
}


__global__ void matrixMultiplicationKernel(double* A, double* B, double* C, int NA, int MA, int NB, int MB) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    double tmpSum = 0;

    if (ROW < NA && COL < MB) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < MA; i++) {
            tmpSum += A[ROW * MA + i] * B[i * MB + COL];
        }
    }
    C[ROW * MB + COL] = tmpSum;
}


__global__ void matrixConvolutionSameKernel(double* A, double* K, double* C, int N, int M, int KN) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    double tmpSum = 0;
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


__global__ void matrixConvolutionValidKernel(double* A, double* K, double* C, int N, int M, int KN) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    double tmpSum = 0;
    if (ROW < N - 2*(KN/2) && COL < M - 2*(KN/2)) {
        // each thread computes one element of the block sub-matrix
        for (int i = KN / 2 * -1; i <= KN / 2; i++) {
            for (int j = KN / 2 * -1; j <= KN / 2; j++) {
                tmpSum += A[(ROW + i + KN/2) * M + COL + j + KN/2] * K[(KN / 2 + i) * KN + j + KN / 2];
            }
        }
        C[ROW * (M - 2*(KN/2)) + COL] = tmpSum;
    }
}



__device__ double relu(double x) {
    return x > 0 ? x : 0;
}

__device__ double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}


__global__ void matrixApplyFunctionKernel(double* A, double* B, int N, int func_id) {
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

__global__ void matrixApplySoftmaxKernel(double* A, double* B, int N, double* sum) {
    int POS = blockIdx.x*blockDim.x+threadIdx.x;
    if (POS < N) {
        B[POS] = B[POS] / (*sum);
    }
}


__global__ void matrixApplySumKernel(double* A, double* res, int N) {

    __shared__ double sdata[256];

// perform first level of reduction,
// reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = A[i] + A[i+blockDim.x];
    __syncthreads();
// do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
// write result for this block to global mem
    if (tid == 0) res[blockIdx.x] = sdata[0];
}


__global__ void matrixAvgPoolingKernel(double *A, double *B, int N, int M, int output_N, int output_M, int stride)  {
    int ROW_out = blockIdx.y*blockDim.y+threadIdx.y;
    int COL_out = blockIdx.x*blockDim.x+threadIdx.x;

    double sum = 0;
    if (ROW_out < output_N && COL_out < output_M) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < stride; i++) {
            for (int j = 0; j < stride; j++) {
                int ROW_in = ROW_out * stride + i;
                int COL_in = COL_out * stride + j;
                if (ROW_in < N && COL_in < M)
                    sum += A[ROW_in*M + COL_in];
            }
        }
        B[ROW_out * output_M + COL_out] = sum / (stride * stride);
    }
}





void matrixAddition(double *A, double *B, double *C, int N, int M) {
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

void matrixMultiplication(double *A, double *B, double *C, int NA, int MA, int NB, int MB){
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

void matrixConvSame(double *A, double *K, double *C, int N, int M, int KN) {
    dim3 threadsPerBlock(M, N);
    dim3 blocksPerGrid(1, 1);
    if (M*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(M)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixConvolutionSameKernel<<<blocksPerGrid,threadsPerBlock>>>(A, K, C, N, M, KN);
}


void matrixConvValid(double *A, double *K, double *C, int N, int M, int KN) {
    dim3 threadsPerBlock(M - 2*(KN/2), N - 2*(KN/2));
    dim3 blocksPerGrid(1, 1);
    if (M*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(M)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixConvolutionValidKernel<<<blocksPerGrid,threadsPerBlock>>>(A, K, C, N, M, KN);
}


void matrixApplyFunction(double* A, double* B, int N, std::string func) {
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


void matrixApplySoftmax(double* A, double* B, int N, double* sum) {
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
    if (N > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }
    matrixApplySoftmaxKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, N, sum);
}


void matrixSum(double* A, double*buff, int N) {
    dim3 threadsPerBlock(N);
    dim3 blocksPerGrid(1);
    if (N > 512) {
        threadsPerBlock.x = 512;
        blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
    }
    matrixApplySumKernel<<<blocksPerGrid,threadsPerBlock>>>(A, buff, N);
}


void matrixAvgPooling(double *A, double *B, int N, int M, int output_N, int output_M, int stride) {
    dim3 threadsPerBlock(output_M, output_N);
    dim3 blocksPerGrid(1, 1);
    if (M*N > 512) {
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(double(M)/double(threadsPerBlock.x));
        blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
    }
    matrixAvgPoolingKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, N, M, output_N, output_M, stride);
}


#include <cmath>

#include "kernel_mat_op.hh"
