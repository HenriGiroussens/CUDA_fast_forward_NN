#include "matrix_add.hh"
#include "matrix_mult.hh"
#include "matrix_conv.hh"


int main() {
    int N = 5;
    int KN = 3;

    std::vector<std::vector<float>> A(N);
    std::vector<std::vector<float>> K(KN);

    A = {{3., 3., 2., 1., 0.},
         {0., 0., 1., 3., 1.},
         {3., 1., 2., 2., 3.},
         {2., 0., 0., 2., 2.},
         {2., 0., 0., 0., 1.}};
    K = {{ 0.,  1., 2. },
         {2.,  2., 0.},
         {0., 1., 2.}};
    mat_conv(A, K, N, KN);
}
