//
// Created by henri on 10/01/2020.
//


#include "Dense.hh"


Dense::Dense(int inputDim, int outputDim, double *weightMatrix, double *biasVector) : input_dim(inputDim),
                                                                                    output_dim(outputDim),
                                                                                    weight_matrix(weightMatrix),
                                                                                    bias_vector(biasVector) {}

double **Dense::forward(double **input) {
    double* output_pre_bias = mat_mult(input[0], weight_matrix, 1, input_dim, input_dim, output_dim);
    double* output = mat_add(output_pre_bias, bias_vector, 1, output_dim, 1, output_dim);
    double** out = &output;
    free(output_pre_bias);
    return out;
}

Dense::Dense(int inputDim, int outputDim) : input_dim(inputDim), output_dim(outputDim) {}

void Dense::setWeightMatrix(double *weightMatrix) {
    weight_matrix = weightMatrix;
}

void Dense::setBiasVector(double *biasVector) {
    bias_vector = biasVector;
}

Dense::~Dense() {
}


