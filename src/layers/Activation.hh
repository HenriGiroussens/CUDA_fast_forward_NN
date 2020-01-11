//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_ACTIVATION_HH
#define GPGPU_ACTIVATION_HH

#include <string>
#include "../matrices_operations/apply_fct.hh"
#include "../matrices_operations/apply_softmax.hh"

class Activation {
private:
    std::string function_name;
    int input_dim;

public:
    Activation(std::string functionName, int inputDim);
    float* forward(float* input);
};

#endif //GPGPU_ACTIVATION_HH