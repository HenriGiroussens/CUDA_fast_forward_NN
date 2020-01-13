//
// Created by henri on 13/01/2020.
//

#include <cstdlib>
#include <cstring>
#include "Flatten.hh"

Flatten::Flatten(int inputN, int inputChannels) : input_dim(inputN),
                                                              input_channels(inputChannels) {}

double **Flatten::forward(double **input) {
    auto **output = static_cast<double **>(malloc(sizeof(double *)));
    auto* out = static_cast<double *>(malloc(input_dim * input_channels * sizeof(double)));
    for (int i = 0; i < input_channels; ++i){
        std::memcpy(&out[i*input_dim], input[i], input_dim* sizeof(double));
        free(input[i]);
    }
    output[0] = out;
    return output;
}
