//
// Created by henri on 11/01/2020.
//

#include "Model.hh"


Model::Model(Layer *layers, int nbLayers, int inputDim, int outputDim) : layers(layers), nb_layers(nbLayers),
                                                                         input_dim(inputDim), output_dim(outputDim) {}



float *Model::predict(float *input) {
    float* tmp = layers[0].forward(input);
    for (int i = 0; i < nb_layers; ++i) {
        tmp = layers[i].forward(tmp);
    }
    return tmp;
}

