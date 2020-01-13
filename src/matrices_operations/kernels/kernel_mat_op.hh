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

void matrixMultiplication(double *A, double *B, double *C, int NA, int MA, int NB, int MB);
void matrixAddition(double *A, double *B, double *C, int N, int M);
void matrixConvSame(double *A, double *K, double *C, int N, int M, int KN);
void matrixConvValid(double *A, double *K, double *C, int N, int M, int KN);
void matrixApplyFunction(double* A, double* B, int N, std::string func);
void matrixSum(double* A, double* buff, int N);
void matrixApplySoftmax(double* A, double* B, int N, double* sum);
void matrixAvgPooling(double *A, double *B, int N, int M, int output_N, int output_M, int stride);
void matrixAddScalar(double *A, double *B, double scalar, int N, int M);

#endif //CMAKE_AND_CUDA_KERNEL_MAT_OP_HH
