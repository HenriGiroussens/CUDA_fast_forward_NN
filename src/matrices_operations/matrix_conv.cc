//
// Created by henri on 09/01/2020.
//

#include "matrix_conv.hh"
#include "kernels/kernel_mat_op.hh"

int mat_conv(std::vector<std::vector<float>> A, std::vector<std::vector<float>> K, int N, int KN) {
    cudaError_t rc = cudaSuccess;

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int SIZE = N*N;
    int SIZE_K = KN*KN;

    // Allocate memory on the host
    std::vector<float> h_A(SIZE);
    std::vector<float> h_K(SIZE_K);
    std::vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = A[i][j];
        }
    }
    for (int i=0; i<KN; i++){
        for (int j=0; j<KN; j++){
            h_K[i*KN+j] = K[i][j];
        }
    }

    // Allocate memory on the device
    float* d_A;
    float* d_K;
    float* d_C;

    cudaMalloc(&d_A, SIZE * sizeof(float));
    cudaMalloc(&d_K, SIZE_K * sizeof(float));
    cudaMalloc(&d_C, SIZE * sizeof(float));

    // Copy to device
    rc = cudaMemcpy(d_A, &h_A[0], SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (rc)
        std::cout << "error memcpy\n";
    cudaMemcpy(d_K, &h_K[0], SIZE_K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, SIZE * sizeof(float));

    // call the kernel
    matrixConv(d_A, d_K, d_C, N, KN);
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

    // check true
    for (int i=0; i<N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << '\n';

    return 0;
}
