//
// Created by henri on 10/01/2020.
//


#include "Dense.hh"


Dense::Dense(int inputDim, int outputDim, float *weightMatrix, float *biasVector) : input_dim(inputDim),
                                                                                    output_dim(outputDim),
                                                                                    weight_matrix(weightMatrix),
                                                                                    bias_vector(biasVector) {}

float* Dense::forward(float *input) {
    float* output_pre_bias = mat_mult(input, weight_matrix, 1, input_dim, input_dim, output_dim);
    float* output = mat_add(output_pre_bias, bias_vector, 1, output_dim, 1, output_dim);
    return output;
}


