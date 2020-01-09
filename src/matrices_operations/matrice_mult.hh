//
// Created by henri on 09/01/2020.
//

#ifndef GPGPU_MATRICE_MULT_HH
#define GPGPU_MATRICE_MULT_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

int mat_mult(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B, int N);

#endif //GPGPU_MATRICE_MULT_HH
