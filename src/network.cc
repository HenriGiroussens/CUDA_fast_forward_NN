//
// Created by henri on 10/01/2020.
//

#include "layers/Model.hh"
#include "layers/Dense.hh"
#include "layers/Activation.hh"

int main() {

    float input[5] = {1., 1., 1., 1., 1.};

    float W1[25] =
            {1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.};
    float B1[5] = {1., 1., 1., 1., 1.};
    float W2[10] =
            {1., 1., 1., 1., 1.,
             1., 1., 1., 1., 1.};
    float B2[2] = {1., 1.};


    Dense dense_1(5, 5, W1, B1);
    Activation act_1("relu", 5);
    Dense dense_2(5, 2, W2, B2);
    Activation act_2("softmax", 2);
    Layer* layers[4] = {&dense_1, &act_1, &dense_2, &act_2};
    Model model(layers, 4, 5, 2);
    float* output = model.predict(input);
    for (int i=0; i<2; i++) {
        std::cout << output[i] << ' ';
    }
    return 0;
}