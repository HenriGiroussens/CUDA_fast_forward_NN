//
// Created by henri on 13/01/2020.
//

#include "matrix_add_scalar.hh"
#include "kernels/kernel_mat_op.hh"

double* mat_add_scalar(double* A, double scalar, int N, int M) {

    int SIZE = N*M;
    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    double* d_A;
    double* d_B;
    auto* B = (double*)malloc(SIZE * sizeof(double));

    cudaMalloc(&d_A, SIZE * sizeof(double));
    cudaMalloc(&d_B, SIZE * sizeof(double));

    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE * sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemset(d_B, 0, SIZE * sizeof(double));

    // call the kernel
    matrixAddScalar(d_A, d_B, scalar, N, M);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&B[0], d_B, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    return B;
}
