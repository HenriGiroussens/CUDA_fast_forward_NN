//
// Created by henri on 11/01/2020.
//

#include <cmath>
#include <utility>

#include "Activation.hh"

Activation::Activation(std::string functionName, int inputDim) : function_name(std::move(functionName)),
                                                                        input_dim(inputDim) {}

float *Activation::forward(float *input) {
    if (function_name != "softmax")
        return apply_fct(input, input_dim, function_name);
    return apply_softmax(input, input_dim);
}

