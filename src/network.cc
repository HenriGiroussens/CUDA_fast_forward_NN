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
        double input_tmp[5] = {1., 1., 1., 1., 1.};
        double* input_1 = input_tmp;
        double** input = &input_1;

        double W1[25] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};
        double B1[5] = {1., 1., 1., 1., 1.};
        double W2[10] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};
        double B2[2] = {1., 1.};


        Dense dense_1(5, 5, W1, B1);
        Activation act_1("relu", 5);
        Dense dense_2(5, 2, W2, B2);
        Activation act_2("softmax", 2);
        Layer *layers[4] = {&dense_1, &act_1, &dense_2, &act_2};
        Model model(layers, 4, 5, 2);
        double *output = model.predict(input);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
    }

    if (mod == "conv") {
        double A_tmp[25] =
                {3., 3., 2., 1., 0.,
                 0., 0., 1., 3., 1.,
                 3., 1., 2., 2., 3.,
                 2., 0., 0., 2., 2,
                 2., 0., 0., 0., 1.};
        double* A_1 = A_tmp;
        double** A = &A_1;

        double K_in [9] = {0., 1., 2.,
                      2., 2., 0.,
                      0., 1., 2.};
        double *K_1 = K_in;
        double **K_2 = &K_1;
        double ***K = &K_2;
        double W2[18] =
                {1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.};
        double B2[2] = {1., 1.};


        Conv2D conv_1(5, 5, 1, K, 1, 3, "valid");
        Activation act_1("relu", 9);
        Dense dense_2(9, 2, W2, B2);
        Activation act_2("softmax", 2);
        Layer *layers[4] = {&conv_1, &act_1, &dense_2, &act_2};
        Model model(layers, 4, 5, 2);
        double *output = model.predict(A);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
        std::cout << '\n';
    }
    return 0;
}