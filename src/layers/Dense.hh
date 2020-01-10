//
// Created by henri on 10/01/2020.
//

#ifndef GPGPU_DENSE_HH
#define GPGPU_DENSE_HH

class Dense {
private:
    int nb_neurone;
    int input_dim;
    float* weight_matrix;
    float* bias_vector;


public:
    Dense(int nbNeurone, int inputDim);

};

#endif //GPGPU_DENSE_HH
