//
// Created by henri on 10/01/2020.
//

#ifndef GPGPU_DENSE_HH
#define GPGPU_DENSE_HH

#include <cstdlib>
#include "../matrices_operations/matrix_add.hh"
#include "../matrices_operations/matrix_mult.hh"

class Dense {
private:
    int input_dim;
    int output_dim;
    float* weight_matrix;
    float* bias_vector;


public:
    Dense(int inputDim, int outputDim, float *weightMatrix, float *biasVector);

    float* forward(float* input);

};

#endif //GPGPU_DENSE_HH
