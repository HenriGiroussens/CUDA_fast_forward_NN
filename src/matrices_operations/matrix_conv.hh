//
// Created by henri on 09/01/2020.
//

#ifndef GPGPU_MATRIX_CONV_HH
#define GPGPU_MATRIX_CONV_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

int mat_conv(std::vector<std::vector<float>> A, std::vector<std::vector<float>> K, int N, int KN);

#endif //GPGPU_MATRIX_CONV_HH
