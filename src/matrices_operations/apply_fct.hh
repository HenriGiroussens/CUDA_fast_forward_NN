//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_APPLY_FCT_HH
#define GPGPU_APPLY_FCT_HH

#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

float* apply_fct(float* A, int N, std::string func);

#endif //GPGPU_APPLY_FCT_HH
