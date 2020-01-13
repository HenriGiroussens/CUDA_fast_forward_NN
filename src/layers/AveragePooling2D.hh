//
// Created by henri on 13/01/2020.
//

#ifndef GPGPU_AVERAGEPOOLING2D_HH
#define GPGPU_AVERAGEPOOLING2D_HH


#include <string>
#include "Layer.hh"

class AveragePooling2D: public Layer {
private:
    int input_N;
    int input_M;
    int input_channels;
    int strides;
    std::string padding;

public:
    AveragePooling2D(int inputN, int inputM, int inputChannels, int strides, const std::string &padding);

    double **forward(double **input) override;
};


#endif //GPGPU_AVERAGEPOOLING2D_HH
