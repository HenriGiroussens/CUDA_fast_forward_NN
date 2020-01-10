//
// Created by henri on 09/01/2020.
//

#ifndef GPGPU_MATRIX_ADD_HH
#define GPGPU_MATRIX_ADD_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

int mat_add(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B, int N);

#endif //GPGPU_MATRIX_ADD_HH
