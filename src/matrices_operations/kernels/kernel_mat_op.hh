//
// Created by henri on 08/01/2020.
//

#ifndef CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
#define CMAKE_AND_CUDA_KERNEL_MAT_OP_HH

#include <cmath>
#include <iostream>
#include "cuda_runtime.h"
#include <cstdlib>
#include "device_launch_parameters.h"

void matrixMultiplication(float *A, float *B, float *C, int NA, int MA, int NB, int MB);
void matrixAddition(float *A, float *B, float *C, int N, int M);
void matrixConv(float *A, float *K, float *C, int N, int M, int KN);
void matrixApplyFunction(float* A, float* B, int N, std::string func);
float matrixSum(float* A, int N);
void matrixApplySoftmax(float* A, float* B, int N, float sum);

#endif //CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
