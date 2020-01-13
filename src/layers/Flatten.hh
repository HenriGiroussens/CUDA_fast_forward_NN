//
// Created by henri on 13/01/2020.
//

#ifndef GPGPU_FLATTEN_HH
#define GPGPU_FLATTEN_HH


#include "Layer.hh"

class Flatten : public Layer {
private:
    int input_dim;
    int input_channels;

public:
    Flatten(int inputN, int inputChannels);
    double** forward(double** input) override;

};


#endif //GPGPU_FLATTEN_HH
