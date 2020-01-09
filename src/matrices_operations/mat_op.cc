#include "matrice_add.hh"
#include "matrice_mult.hh"


int main() {
    int N = 20;

    std::vector<std::vector<float>> A(N);
    std::vector<std::vector<float>> B(N);
    for (int i = 0; i < N; i++) {
        std::vector<float> a(N);
        std::vector<float> b(N);
        A[i] = a;
        B[i] = b;
        for (int j = 0; j < N; j++) {
            A[i][j] = sin(i) * sin(i);
            B[i][j] = cos(i) * cos(i);
        }
    }
    mat_mult(A, B, N);
}
