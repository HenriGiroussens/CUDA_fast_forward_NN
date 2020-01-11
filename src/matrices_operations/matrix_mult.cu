//
// Created by henri on 09/01/2020.
//


#include "kernels/kernel_mat_op.hh"



float* mat_mult(float* A, float* B, int NA, int MA, int NB, int MB)
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
    float* d_A;
    float* d_B;
    float* d_C;
    auto* C = (float*)malloc(SIZE_C * sizeof(float));

    cudaMalloc(&d_A, SIZE_A * sizeof(float));
    cudaMalloc(&d_B, SIZE_B * sizeof(float));
    cudaMalloc(&d_C, SIZE_C * sizeof(float));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE_A * sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &B[0], SIZE_B * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE_C * sizeof(float));

    // call the kernel
    matrixMultiplication(d_A, d_B, d_C, NA, MA, NB, MB);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&C[0], d_C, SIZE_C * sizeof(float), cudaMemcpyDeviceToHost);

    return C;
}