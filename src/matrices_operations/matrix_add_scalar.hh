//
// Created by henri on 13/01/2020.
//

#ifndef GPGPU_MATRIX_ADD_SCALAR_HH
#define GPGPU_MATRIX_ADD_SCALAR_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

double* mat_add_scalar(double* A, double scalar, int N, int M);

#endif //GPGPU_MATRIX_ADD_SCALAR_HH
