//
// Created by henri on 11/01/2020.
//

#include <utility>
#include "kernels/kernel_mat_op.hh"


double* apply_fct(double* A, int N, std::string func) {

    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    double* d_A;
    double* d_B;
    auto* B = (double*)malloc(N * sizeof(double));

    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], N* sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemset(d_B, 0, N * sizeof(double));

    // call the kernel
    matrixApplyFunction(d_A, d_B, N, std::move(func));
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&B[0], d_B, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    return B;
}
