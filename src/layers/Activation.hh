//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_ACTIVATION_HH
#define GPGPU_ACTIVATION_HH

#include <string>
#include "../matrices_operations/apply_fct.hh"
#include "../matrices_operations/apply_softmax.hh"
#include "Layer.hh"

class Activation : public Layer {
private:
    std::string function_name;
    int input_dim;
    int input_channels;

public:
    Activation(const std::string &functionName, int inputDim, int inputChannels);

    double** forward(double** input) override;
};

#endif //GPGPU_ACTIVATION_HH
