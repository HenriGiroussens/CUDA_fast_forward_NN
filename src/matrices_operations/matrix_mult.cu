//
// Created by henri on 09/01/2020.
//

#include "matrix_mult.hh"
#include "kernels/kernel_mat_op.hh"



int mat_mult(std::vector<std::vector<float>> A, std::vector<std::vector<float>> B, int N)
{
    cudaError_t rc = cudaSuccess;

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int SIZE = N*N;

    // Allocate memory on the host
    std::vector<float> h_A(SIZE);
    std::vector<float> h_B(SIZE);
    std::vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = A[i][j];
            h_B[i*N+j] = B[i][j];
        }
    }

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;

    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_B, SIZE * sizeof(float));
    cudaMalloc(&d_C, SIZE * sizeof(float));

    // Copy to device
    rc = cudaMemcpy(d_A, &h_A[0], SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_B, &h_B[0], SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE * sizeof(float));

    // call the kernel
    matrixMultiplication(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // copy memory back to host
    cudaMemcpy(&h_C[0], d_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    //un-flatten matrix
    std::vector<std::vector<float>> C(N);

    for (int i=0; i<N; i++) {
        std::vector<float> c(N);
        C[i] = c;
        for (int j = 0; j < N; j++) {
            C[i][j] = h_C[i * N + j];
        }
    }


    // compute CPU mult
    std::vector<std::vector<float>> C_true(N);
    float sum;
    for (int i=0; i<N; i++) {
        std::vector<float> c_true(N);
        C_true[i] = c_true;
        for (int j = 0; j < N; j++) {
            sum = 0.f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i*N+k] * h_B[k*N+j];
            }
            C_true[i][j] = sum;
        }
    }
    bool same = true;
    // check true
    for (int i=0; i<N; i++) {
        for (int j = 0; j < N; j++) {
            if (C[i][j] < C_true[i][j] - 0.0001 || C[i][j] > C_true[i][j] + 0.0001)
                same = false;
            std::cout << C[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

    for (int i=0; i<N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C_true[i][j] << ' ';
        }
        std::cout << '\n';
    }
    if (same)
        std::cout << "succes ! \n";
    else
        std::cout << "failure ! \n";

    return 0;
}