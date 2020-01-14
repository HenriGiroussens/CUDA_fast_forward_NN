//
// Created by henri on 11/01/2020.
//

#ifndef GPGPU_MODEL_HH
#define GPGPU_MODEL_HH


#include <vector>
#include "Layer.hh"

class Model {
private:
    std::vector<Layer*> layers;
    int nb_layers;
    int input_dim;
    int output_dim;

public:
    Model(std::vector<Layer*> layers, int nbLayers, int inputDim, int outputDim);

    double* predict(double** input);
};


#endif //GPGPU_MODEL_HH
