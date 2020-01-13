//
// Created by henri on 12/01/2020.
//

#ifndef GPGPU_CONV2D_HH
#define GPGPU_CONV2D_HH


#include <string>
#include "Layer.hh"

class Conv2D : public Layer {
private:
    int input_N;
    int input_M;
    int input_channels;
    double*** kernels;
    double* bias_vector;
    int output_channels;
    int kernel_size;
    std::string padding;


public:
    Conv2D(int inputN, int inputM, int inputChannels, double ***kernels, double *biasVector, int outputChannels,
           int kernelSize, const std::string &padding);

    double** forward(double** input) override;
};


#endif //GPGPU_CONV2D_HH
