//
// Created by henri on 11/01/2020.
//

#include <utility>
#include "kernels/kernel_mat_op.hh"


float* apply_fct(float* A, int N, std::string func) {

    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    float* d_A;
    float* d_B;
    auto* B = (float*)malloc(N * sizeof(float));

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], N* sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemset(d_B, 0, N * sizeof(float));

    // call the kernel
    matrixApplyFunction(d_A, d_B, N, std::move(func));
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&B[0], d_B, N * sizeof(float), cudaMemcpyDeviceToHost);

    return B;
}
