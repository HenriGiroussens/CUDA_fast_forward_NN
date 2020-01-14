//
// Created by henri on 09/01/2020.
//

#include "matrix_conv.hh"
#include "kernels/kernel_mat_op.hh"

double* mat_conv(double* A, double* K, int NA, int MA, int NK, bool padding_valid) {

    if (NK % 2 == 0) {
        std::cerr << "shape error" << std::endl;
        return nullptr;
    }
    cudaError_t rc = cudaSuccess;

    int SIZE_A = NA*MA;
    int SIZE_K = NK*NK;



    // Allocate memory on the device
    double* d_A;
    double* d_B;
    double* d_C;

    cudaMalloc(&d_A, SIZE_A * sizeof(double));
    cudaMalloc(&d_B, SIZE_K * sizeof(double));


    // Copy to device
    rc = cudaMemcpy(d_A, &A[0], SIZE_A * sizeof(double), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &K[0], SIZE_K * sizeof(double), cudaMemcpyHostToDevice);

    if (!padding_valid) {
        int SIZE_C = SIZE_A;
        auto *C = (double *) malloc(SIZE_C * sizeof(double));
        cudaMalloc(&d_C, SIZE_C * sizeof(double));
        cudaMemset(d_C, 0, SIZE_C * sizeof(double));

        // call the kernel
        matrixConvSame(d_A, d_B, d_C, NA, MA, NK);
        cudaDeviceSynchronize();

        // copy memory back to host
        cudaMemcpy(&C[0], d_C, SIZE_C * sizeof(double), cudaMemcpyDeviceToHost);

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return C;
    }

    else {
        int SIZE_C = (NA - 2*(NK/2)) * (MA - 2*(NK/2));
        auto *C = (double *) malloc(SIZE_C * sizeof(double));
        rc = cudaMalloc(&d_C, SIZE_C * sizeof(double));
        if (rc)
            std::cout << "error malloc\n";
        rc = cudaMemset(d_C, 0, SIZE_C * sizeof(double));
        if (rc)
            std::cout << "error memset\n";

        // call the kernel
        matrixConvValid(d_A, d_B, d_C, NA, MA, NK);
        cudaDeviceSynchronize();

        // copy memory back to host
        rc = cudaMemcpy(&C[0], d_C, SIZE_C * sizeof(double), cudaMemcpyDeviceToHost);
        if (rc)
            std::cout << "error memcpy\n";

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return C;
    }
}
