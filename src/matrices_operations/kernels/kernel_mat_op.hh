//
// Created by henri on 08/01/2020.
//

#ifndef CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
#define CMAKE_AND_CUDA_KERNEL_MAT_OP_HH

void matrixMultiplication(float *A, float *B, float *C, int N);
void matrixAddition(float *A, float *B, float *C, int N);
void matrixConv(float *A, float *K, float *C, int N, int KN);

#endif //CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
