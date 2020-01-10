//
// Created by henri on 08/01/2020.
//

#ifndef CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
#define CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
void matrixMultiplication(float *A, float *B, float *C, int NA, int MA, int NB, int MB);
void matrixAddition(float *A, float *B, float *C, int N, int M);
void matrixConv(float *A, float *K, float *C, int N, int M, int KN);

#endif //CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
