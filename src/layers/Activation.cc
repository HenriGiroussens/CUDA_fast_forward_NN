//
// Created by henri on 11/01/2020.
//

#include <cmath>
#include <utility>

#include "Activation.hh"

double **Activation::forward(double **input) {
    auto** out = static_cast<double **>(malloc(input_channels * sizeof(double *)));
    for (int i = 0; i < input_channels; ++i) {
        double *output;
        if (function_name != "softmax")
            output = apply_fct(input[i], input_dim, function_name);
        else
            output = apply_softmax(input[i], input_dim);
        out[i] = output;
        free(input[i]);
    }
    return out;
}

Activation::Activation(const std::string &functionName, int inputDim, int inputChannels) : function_name(functionName),
                                                                                           input_dim(inputDim),
                                                                                           input_channels(
                                                                                                   inputChannels) {}

