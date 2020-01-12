//
// Created by henri on 11/01/2020.
//

#include "apply_softmax.hh"
#include "kernels/kernel_mat_op.hh"

double* apply_softmax(double* A, int N) {
    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    double* d_A;
    double* d_B;
    double* buff;
    auto* B = (double*)malloc(N * sizeof(double));

    cudaMalloc(&d_A, N * sizeof(double));
    cudaMalloc(&d_B, N * sizeof(double));
    cudaMalloc(&buff, sizeof(double));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], N* sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemset(d_B, 0, N * sizeof(double));
    cudaMemset(buff, 0, sizeof(double));

    // call the kernel
    matrixApplyFunction(d_A, d_B, N, "exp");
    cudaDeviceSynchronize();
    matrixSum(d_B, buff, N);
    cudaDeviceSynchronize();
    matrixApplySoftmax(d_A, d_B, N, buff);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&B[0], d_B, N * sizeof(double), cudaMemcpyDeviceToHost);

    return B;
}