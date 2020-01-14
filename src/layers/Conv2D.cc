//
// Created by henri on 12/01/2020.
//

#include <utility>
#include <cstring>

#include "Conv2D.hh"
#include "../matrices_operations/matrix_conv.hh"
#include "../matrices_operations/matrix_add.hh"
#include "../matrices_operations/matrix_add_scalar.hh"

Conv2D::Conv2D(int inputN, int inputM, int inputChannels, double ***kernels, double *biasVector, int outputChannels,
               int kernelSize, const bool &padding) : input_N(inputN), input_M(inputM),
                                                             input_channels(inputChannels), kernels(kernels),
                                                             bias_vector(biasVector), output_channels(outputChannels),
                                                             kernel_size(kernelSize), padding_valid(padding) {}


double **Conv2D::forward(double **input) {
    auto** output = static_cast<double **>(malloc(output_channels * sizeof(double *)));
    for (int i = 0; i < output_channels; ++i) {
        double *partial_output_all_channels;
        for (int j = 0; j < input_channels; ++j) {
            double *partial_output_single_channel = mat_conv(input[j], kernels[i][j], input_N, input_M, kernel_size, padding_valid);
            if (j == 0) {
                if (!padding_valid) {
                    partial_output_all_channels = static_cast<double *>(malloc(input_N * input_M * sizeof(double)));
                    std::memcpy(partial_output_all_channels, partial_output_single_channel, input_N * input_M * sizeof(double));
                }
                else {
                    partial_output_all_channels = static_cast<double *>(malloc((input_N - kernel_size + 1) * (input_M - kernel_size + 1) * sizeof(double)));
                    std::memcpy(partial_output_all_channels, partial_output_single_channel, (input_N - kernel_size + 1) * (input_M - kernel_size + 1) * sizeof(double));
                }
            }
            else {
                if (!padding_valid) {
                    partial_output_all_channels = mat_add(partial_output_all_channels, partial_output_single_channel,
                                                          input_N, input_M, input_N, input_M);
                }
                else
                    partial_output_all_channels = mat_add(partial_output_all_channels, partial_output_single_channel, input_N - kernel_size + 1, input_M - kernel_size + 1, input_N - kernel_size + 1, input_M - kernel_size + 1);
            }
            free(partial_output_single_channel);
        }
        partial_output_all_channels = mat_add_scalar(partial_output_all_channels, bias_vector[i], input_N, input_M);
        output[i] = partial_output_all_channels;
    }
    return output;
}

Conv2D::Conv2D(int inputN, int inputM, int inputChannels, int outputChannels, int kernelSize,
               const bool &padding) : input_N(inputN), input_M(inputM), input_channels(inputChannels),
                                             output_channels(outputChannels), kernel_size(kernelSize),
                                             padding_valid(padding) {}

void Conv2D::setKernels(double ***kernels) {
    Conv2D::kernels = kernels;
}

void Conv2D::setBiasVector(double *biasVector) {
    bias_vector = biasVector;
}






