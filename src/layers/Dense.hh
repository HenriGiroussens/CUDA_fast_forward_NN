//
// Created by henri on 10/01/2020.
//

#ifndef GPGPU_DENSE_HH
#define GPGPU_DENSE_HH

#include <cstdlib>
#include "../matrices_operations/matrix_add.hh"
#include "../matrices_operations/matrix_mult.hh"
#include "Layer.hh"

class Dense : public Layer {
private:
    int input_dim;
    int output_dim;
    double* weight_matrix;
    double* bias_vector;


public:
    Dense(int inputDim, int outputDim, double *weightMatrix, double *biasVector);

    double** forward(double** input) override;


};

#endif //GPGPU_DENSE_HH
