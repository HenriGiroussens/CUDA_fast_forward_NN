//
// Created by henri on 11/01/2020.
//

#include <cmath>
#include <utility>

#include "Activation.hh"

Activation::Activation(std::string functionName, int inputDim) : function_name(std::move(functionName)),
                                                                        input_dim(inputDim) {}


double **Activation::forward(double **input) {
    double* output;
    if (function_name != "softmax")
        output = apply_fct(input[0], input_dim, function_name);
    else
        output = apply_softmax(input[0], input_dim);
    double** out = &output;
    return out;
}

