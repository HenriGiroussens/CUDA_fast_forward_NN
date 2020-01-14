//
// Created by henri on 09/01/2020.
//

#ifndef GPGPU_MATRIX_CONV_HH
#define GPGPU_MATRIX_CONV_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

double* mat_conv(double* A, double* K, int NA, int MA, int NK, bool padding);

#endif //GPGPU_MATRIX_CONV_HH
