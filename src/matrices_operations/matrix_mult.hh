//
// Created by henri on 09/01/2020.
//

#ifndef GPGPU_MATRIX_MULT_HH
#define GPGPU_MATRIX_MULT_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

float* mat_mult(float* A, float* B, int NA, int MA, int NB, int MB);

#endif //GPGPU_MATRIX_MULT_HH
