//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_APPLY_SOFTMAX_HH
#define GPGPU_APPLY_SOFTMAX_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

float* apply_softmax(float* A, int N);

#endif //GPGPU_APPLY_SOFTMAX_HH
