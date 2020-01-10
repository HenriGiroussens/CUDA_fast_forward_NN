//
// Created by henri on 09/01/2020.
//

#include "matrix_add.hh"

#include "kernels/kernel_mat_op.hh"


float* mat_add(float* A, float* B, int NA, int MA, int NB, int MB)
{
    if (NA != NB && MA != MB) {
        std::cerr << "shape error" << std::endl;
        return nullptr;
    }

    int SIZE = NA*MA;
    cudaError_t rc = cudaSuccess;
    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;
    auto* C = (float*)malloc(SIZE * sizeof(float));

    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_C, SIZE * sizeof(float));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &B[0], SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE * sizeof(float));

    // call the kernel
    matrixAddition(d_A, d_B, d_C, NA, MA);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&C[0], d_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    return C;
}