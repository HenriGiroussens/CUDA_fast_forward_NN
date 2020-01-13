//
// Created by henri on 10/01/2020.
//

#include "layers/Model.hh"
#include "layers/Dense.hh"
#include "layers/Activation.hh"
#include "layers/Conv2D.hh"
#include "layers/Flatten.hh"
#include "layers/AveragePooling2D.hh"

int main() {

    std::string mod = "small_LeNet";

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
        Activation act_1("relu", 5, 1);
        Dense dense_2(5, 2, W2, B2);
        Activation act_2("softmax", 2, 1);
        Layer *layers[4] = {&dense_1, &act_1, &dense_2, &act_2};
        Model model(layers, 4, 5, 2);
        double *output = model.predict(input);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
    }

    if (mod == "conv") {
        double A_tmp[2][25] = {
                {3., 3., 2., 1., 0.,
                        0., 0., 1., 3., 1.,
                        3., 1., 2., 2., 3.,
                        2., 0., 0., 2., 2,
                        2., 0., 0., 0., 1.},
                {3., 3., 2., 1., 0.,
                        0., 0., 1., 3., 1.,
                        3., 1., 2., 2., 3.,
                        2., 0., 0., 2., 2,
                        2., 0., 0., 0., 1.}
        };
        auto** A = static_cast<double **>(malloc(2 * sizeof(double *)));
        for (int i = 0; i < 2; ++i){
            auto* A_in = static_cast<double *>(malloc(25 * sizeof(double)));
            A[i] = A_in;
            for (int j = 0; j < 25; ++j)
                A[i][j] = A_tmp[i][j];
        }

        double K_tmp [2][2][9] = {{{0., 1., 2.,
                                           2., 2., 0.,
                                           0., 1., 2.},
                                          {0., 1., 2.,
                                                  2., 2., 0.,
                                                  0., 1., 2.}
                                  },
                                  {{0., 1., 2.,
                                           2., 2., 0.,
                                           0., 1., 2.},
                                          {0., 1., 2.,
                                                  2., 2., 0.,
                                                  0., 1., 2.}
                                  }
        };
        auto*** K = static_cast<double ***>(malloc(2 * sizeof(double **)));
        for (int i = 0; i < 2; ++i){
            auto** K1 = static_cast<double **>(malloc(2 * sizeof(double*)));
            for (int j = 0; j < 2; ++j) {
                auto* K2 = static_cast<double *>(malloc(25 * sizeof(double)));
                for (int k = 0; k < 25; ++k)
                    K2[k] = K_tmp[i][j][k];
                K1[j] = K2;
            }
            K[i] = K1;
        }
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


        Conv2D conv_1(5, 5, 2, K, 2, 3, "valid");
        Activation act_1("relu", 9, 2);
        Dense dense_2(9, 2, W2, B2);
        Activation act_2("softmax", 2, 1);
        Layer *layers[4] = {&conv_1, &act_1, &dense_2, &act_2};
        Model model(layers, 4, 5, 2);
        double *output = model.predict(A);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
        std::cout << '\n';
    }

    if (mod == "conv_flatten") {
        double A_tmp[2][25] = {
                {3., 3., 2., 1., 0.,
                        0., 0., 1., 3., 1.,
                        3., 1., 2., 2., 3.,
                        2., 0., 0., 2., 2,
                        2., 0., 0., 0., 1.},
                {3., 3., 2., 1., 0.,
                        0., 0., 1., 3., 1.,
                        3., 1., 2., 2., 3.,
                        2., 0., 0., 2., 2,
                        2., 0., 0., 0., 1.}
        };
        auto** A = static_cast<double **>(malloc(2 * sizeof(double *)));
        for (int i = 0; i < 2; ++i){
            auto* A_in = static_cast<double *>(malloc(25 * sizeof(double)));
            A[i] = A_in;
            for (int j = 0; j < 25; ++j)
                A[i][j] = A_tmp[i][j];
        }

        double K_tmp [2][2][9] = {{{0., 1., 2.,
                                           2., 2., 0.,
                                           0., 1., 2.},
                                          {0., 1., 2.,
                                                  2., 2., 0.,
                                                  0., 1., 2.}
                                  },
                                  {{0., 1., 2.,
                                           2., 2., 0.,
                                           0., 1., 2.},
                                          {0., 1., 2.,
                                                  2., 2., 0.,
                                                  0., 1., 2.}
                                  }
        };
        auto*** K = static_cast<double ***>(malloc(2 * sizeof(double **)));
        for (int i = 0; i < 2; ++i){
            auto** K1 = static_cast<double **>(malloc(2 * sizeof(double*)));
            for (int j = 0; j < 2; ++j) {
                auto* K2 = static_cast<double *>(malloc(25 * sizeof(double)));
                for (int k = 0; k < 25; ++k)
                    K2[k] = K_tmp[i][j][k];
                K1[j] = K2;
            }
            K[i] = K1;
        }
        double W2[36] =
                {1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.};
        double B2[2] = {1., 1.};


        Conv2D conv_1(5, 5, 2, K, 2, 3, "valid");
        Activation act_1("relu", 9, 2);
        Flatten flatten(9, 2);
        Dense dense_2(18, 2, W2, B2);
        Activation act_2("softmax", 2, 1);
        Layer *layers[5] = {&conv_1, &act_1, &flatten, &dense_2, &act_2};
        Model model(layers, 5, 5, 2);
        double *output = model.predict(A);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
        std::cout << '\n';
    }


    if (mod == "small_LeNet") {
        double A_tmp[2][25] = {
                {3., 3., 2., 1., 0.,
                        0., 0., 1., 3., 1.,
                        3., 1., 2., 2., 3.,
                        2., 0., 0., 2., 2,
                        2., 0., 0., 0., 1.},
                {3., 3., 2., 1., 0.,
                        0., 0., 1., 3., 1.,
                        3., 1., 2., 2., 3.,
                        2., 0., 0., 2., 2,
                        2., 0., 0., 0., 1.}
        };
        auto** A = static_cast<double **>(malloc(2 * sizeof(double *)));
        for (int i = 0; i < 2; ++i){
            auto* A_in = static_cast<double *>(malloc(25 * sizeof(double)));
            A[i] = A_in;
            for (int j = 0; j < 25; ++j)
                A[i][j] = A_tmp[i][j];
        }

        double K_tmp [2][2][9] = {{{0., 1., 2.,
                                           2., 2., 0.,
                                           0., 1., 2.},
                                          {0., 1., 2.,
                                                  2., 2., 0.,
                                                  0., 1., 2.}
                                  },
                                  {{0., 1., 2.,
                                           2., 2., 0.,
                                           0., 1., 2.},
                                          {0., 1., 2.,
                                                  2., 2., 0.,
                                                  0., 1., 2.}
                                  }
        };
        auto*** K = static_cast<double ***>(malloc(2 * sizeof(double **)));
        for (int i = 0; i < 2; ++i){
            auto** K1 = static_cast<double **>(malloc(2 * sizeof(double*)));
            for (int j = 0; j < 2; ++j) {
                auto* K2 = static_cast<double *>(malloc(25 * sizeof(double)));
                for (int k = 0; k < 25; ++k)
                    K2[k] = K_tmp[i][j][k];
                K1[j] = K2;
            }
            K[i] = K1;
        }
        double W2[16] =
                {1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.};
        double B2[2] = {1., 1.};


        Conv2D conv_1(5, 5, 2, K, 2, 3, "same");
        Activation act_1("relu", 25, 2);
        AveragePooling2D avg_1(5, 5, 2, 2, "valid");
        Flatten flatten(4, 2);
        Dense dense_2(8, 2, W2, B2);
        Activation act_2("softmax", 2, 1);
        Layer *layers[6] = {&conv_1, &act_1, &avg_1, &flatten, &dense_2, &act_2};
        Model model(layers, 6, 5, 2);
        double *output = model.predict(A);
        for (int i = 0; i < 2; i++) {
            std::cout << output[i] << ' ';
        }
        std::cout << '\n';
    }


    return 0;
}