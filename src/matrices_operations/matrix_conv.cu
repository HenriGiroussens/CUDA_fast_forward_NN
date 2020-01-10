//
// Created by henri on 09/01/2020.
//

#include "matrix_conv.hh"
#include "kernels/kernel_mat_op.hh"

float* mat_conv(float* A, float* K, int NA, int MA, int NK) {

    if (NK % 2 == 0) {
        std::cerr << "shape error" << std::endl;
        return nullptr;
    }
    cudaError_t rc = cudaSuccess;

    int SIZE_A = NA*MA;
    int SIZE_K = NK*NK;
    int SIZE_C = SIZE_A;


    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;
    auto* C = (float*)malloc(SIZE_C * sizeof(float));

    cudaMalloc(&d_A, SIZE_A * sizeof(float));
    cudaMalloc(&d_B, SIZE_K * sizeof(float));
    cudaMalloc(&d_C, SIZE_C * sizeof(float));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE_A * sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &K[0], SIZE_K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE_C * sizeof(float));

    // call the kernel
    matrixConv(d_A, d_B, d_C, NA, MA, NK);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&C[0], d_C, SIZE_C * sizeof(float), cudaMemcpyDeviceToHost);

    return C;
}
