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
    double* kernel;
    int kernel_size;
    std::string padding;

public:
    Conv2D(int inputN, int inputM, double *kernel, int kernelSize, std::string padding);

    double* forward(double* input) override;
};


#endif //GPGPU_CONV2D_HH
