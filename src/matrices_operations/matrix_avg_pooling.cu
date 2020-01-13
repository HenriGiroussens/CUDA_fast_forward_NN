//
// Created by henri on 13/01/2020.
//

#include "matrix_avg_pooling.hh"
#include "kernels/kernel_mat_op.hh"

double* avg_pooling_2D(double* A, int N, int M, int strides, std::string padding) {
    int output_N = N / strides;
    int output_M = M / strides;
    if (padding == "same") {
        if (N%strides != 0)
            output_N++;
        if (M%strides != 0)
            output_M++;
    }
    int SIZE = output_N * output_M;
    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    double* d_A;
    double* d_B;
    auto* B = (double*)malloc(SIZE * sizeof(double));

    cudaMalloc(&d_A, N*M * sizeof(double));
    cudaMalloc(&d_B, SIZE * sizeof(double));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], N*M* sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemset(d_B, 0, SIZE * sizeof(double));

    // call the kernel
    matrixAvgPooling(d_A, d_B, N, M, output_N, output_M, strides);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&B[0], d_B, N * sizeof(double), cudaMemcpyDeviceToHost);

    return B;

}