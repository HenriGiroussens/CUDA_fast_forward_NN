//
// Created by henri on 13/01/2020.
//

#ifndef GPGPU_MATRIX_AVG_POOLING_HH
#define GPGPU_MATRIX_AVG_POOLING_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

double* avg_pooling_2D(double* A, int N, int M, int strides, bool padding);

#endif //GPGPU_MATRIX_AVG_POOLING_HH
