//
// Created by henri on 12/01/2020.
//

#include <utility>

#include "Conv2D.hh"
#include "../matrices_operations/matrix_conv.hh"

Conv2D::Conv2D(int inputN, int inputM, double *kernel, int kernelSize, std::string padding) : input_N(inputN),
                                                                                                    input_M(inputM),
                                                                                                    kernel(kernel),
                                                                                                    kernel_size(
                                                                                                            kernelSize),
                                                                                                    padding(std::move(padding)) {}

double *Conv2D::forward(double *input) {
    double* output = mat_conv(input, kernel, input_N, input_M, kernel_size, padding);
    return output;
}
