//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_MODEL_HH
#define GPGPU_MODEL_HH


#include "Layer.hh"

class Model {
private:
    Layer** layers;
    int nb_layers;
    int input_dim;
    int output_dim;

public:
    Model(Layer **layers, int nbLayers, int inputDim, int outputDim);

    double* predict(double** input);
};


#endif //GPGPU_MODEL_HH
