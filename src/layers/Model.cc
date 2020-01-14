//
// Created by henri on 11/01/2020.
//

#include <utility>

#include "Model.hh"


Model::Model(std::vector<Layer*> layers, int nbLayers, int inputDim, int outputDim) : layers(std::move(layers)), nb_layers(nbLayers),
                                                                         input_dim(inputDim), output_dim(outputDim) {}



double *Model::predict(double **input) {
    double** tmp = layers[0]->forward(input);
    for (int i = 1; i < nb_layers; ++i) {
        tmp = layers[i]->forward(tmp);
    }
    return tmp[0];
}

