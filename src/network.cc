//
// Created by henri on 10/01/2020.
//

#include "layers/Model.hh"
#include "layers/Dense.hh"
#include "layers/Activation.hh"
#include "layers/Conv2D.hh"

int main() {

    std::string mod = "conv";

    if (mod == "dense") {
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
        Layer *layers[4] = {&dense_1, &act_1, &dense_2, &act_2};
        Model model(layers, 4, 5, 2);
        float *output = model.predict(input);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
    }

    if (mod == "conv") {
        float A[25] =
                {3., 3., 2., 1., 0.,
                 0., 0., 1., 3., 1.,
                 3., 1., 2., 2., 3.,
                 2., 0., 0., 2., 2,
                 2., 0., 0., 0., 1.};

        float K[9] = {0., 1., 2.,
                      2., 2., 0.,
                      0., 1., 2.};
        float W2[18] =
                {1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.};
        float B2[2] = {1., 1.};


        Conv2D conv_1(5, 5, K, 3, "valid");
        Activation act_1("relu", 9);
        Dense dense_2(9, 2, W2, B2);
        Activation act_2("softmax", 2);
        Layer *layers[4] = {&conv_1, &act_1, &dense_2, &act_2};
        Model model(layers, 4, 5, 2);
        float *output = model.predict(A);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
        std::cout << '\n';
    }
    return 0;
}