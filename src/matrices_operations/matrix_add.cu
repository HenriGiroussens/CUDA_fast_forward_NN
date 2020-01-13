//
// Created by henri on 09/01/2020.
//

#include "matrix_add.hh"

#include "kernels/kernel_mat_op.hh"


double* mat_add(double* A, double* B, int NA, int MA, int NB, int MB)
{
    if (NA != NB && MA != MB) {
        std::cerr << "shape error" << std::endl;
        return nullptr;
    }

    int SIZE = NA*MA;
    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    double* d_A;
    double* d_B;
    double* d_C;
    auto* C = (double*)malloc(SIZE * sizeof(double));

    cudaMalloc(&d_A, SIZE * sizeof(double));
    cudaMalloc(&d_B, SIZE * sizeof(double));
    cudaMalloc(&d_C, SIZE * sizeof(double));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE * sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &B[0], SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE * sizeof(double));

    // call the kernel
    matrixAddition(d_A, d_B, d_C, NA, MA);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&C[0], d_C, SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}