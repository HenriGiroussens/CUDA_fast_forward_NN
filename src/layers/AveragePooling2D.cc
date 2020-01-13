//
// Created by henri on 13/01/2020.
//

#include <cmath>
#include "AveragePooling2D.hh"
#include "../matrices_operations/matrix_avg_pooling.hh"

AveragePooling2D::AveragePooling2D(int inputN, int inputM, int inputChannels, int strides, const std::string &padding)
        : input_N(inputN), input_M(inputM), input_channels(inputChannels), strides(strides), padding(padding) {}

double **AveragePooling2D::forward(double **input) {
    auto** out = static_cast<double **>(malloc(input_channels * sizeof(double *)));
    for (int i = 0; i < input_channels; ++i) {
        double *output;
        output = avg_pooling_2D(input[i], input_N, input_M, strides, padding);
        out[i] = output;
        free(input[i]);
    }
    return out;
}
