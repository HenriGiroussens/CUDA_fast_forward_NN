//
// Created by henri on 09/01/2020.
//


#include "kernels/kernel_mat_op.hh"



double* mat_mult(double* A, double* B, int NA, int MA, int NB, int MB)
{
    if (MA != NB) {
        std::cerr << "shape error" << std::endl;
        return nullptr;
    }
    cudaError_t rc = cudaSuccess;

    int SIZE_A = NA*MA;
    int SIZE_B = NB*MB;
    int SIZE_C = NA*MB;


    // Allocate memory on the device
    double* d_A;
    double* d_B;
    double* d_C;
    auto* C = (double*)malloc(SIZE_C * sizeof(double));

    cudaMalloc(&d_A, SIZE_A * sizeof(double));
    cudaMalloc(&d_B, SIZE_B * sizeof(double));
    cudaMalloc(&d_C, SIZE_C * sizeof(double));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE_A * sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &B[0], SIZE_B * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE_C * sizeof(double));

    // call the kernel
    matrixMultiplication(d_A, d_B, d_C, NA, MA, NB, MB);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&C[0], d_C, SIZE_C * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}