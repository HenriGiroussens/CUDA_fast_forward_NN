//
// Created by henri on 09/01/2020.
//

#ifndef GPGPU_MATRIX_CONV_HH
#define GPGPU_MATRIX_CONV_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

float* mat_conv(float* A, float* K, int NA, int MA, int NK);

#endif //GPGPU_MATRIX_CONV_HH
