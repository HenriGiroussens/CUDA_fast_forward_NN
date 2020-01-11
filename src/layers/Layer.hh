//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_LAYER_HH
#define GPGPU_LAYER_HH


class Layer {

public:
    virtual float* forward(float *input) = 0;
};


#endif //GPGPU_LAYER_HH
