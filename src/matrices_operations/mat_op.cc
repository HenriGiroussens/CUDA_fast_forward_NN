#include "matrix_add.hh"
#include "matrix_mult.hh"
#include "matrix_conv.hh"
#include "apply_fct.hh"
#include "apply_softmax.hh"


int main() {
    std::string type("softmax");

    if (type=="add") {
        int NA = 4;
        int MA = 5;
        int NB = 4;
        int MB = 5;
        float A[20] =
                {3., 3., 2., 1., 0.,
                0., 0., 1., 3., 1.,
                3., 1., 2., 2., 3.,
                2., 0., 0., 2., 2.};
        float B[20] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};
        float* C = mat_add(A, B, NA, MA, NB, MB);
        for (int i=0; i<NA; i++) {
            for (int j = 0; j < MA; j++) {
                std::cout << C[i * MA + j] << ' ';
            }
            std::cout << '\n';
        }
    }

    if (type=="mult") {
        int NA = 2;
        int MA = 3;
        int NB = 3;
        int MB = 4;
        float A[6] =
                {3., 3., 2.,
                 0., 0., 1.};
        float B[12] =
                {1., 1., 1., 1.,
                 1., 1., 1., 1.,
                 1., 1., 1., 1.};
        float* C = mat_mult(A, B, NA, MA, NB, MB);
        for (int i=0; i<NA; i++) {
            for (int j = 0; j < MB; j++) {
                std::cout << C[i * MB + j] << ' ';
            }
            std::cout << '\n';
        }
    }


    if (type=="conv") {
        int NA = 4;
        int MA = 5;
        int KN = 3;
        float A[20] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};

        float K[9] = {1., 10., 100.,
                      1000., 10000., 100000.,
                      1000000., 10000000., 100000000.};
        float* C =mat_conv(A, K, NA, MA, KN);
        for (int i=0; i<NA; i++) {
            for (int j = 0; j < MA; j++) {
                std::cout << C[i * MA + j] << ' ';
            }
            std::cout << '\n';
        }
    }

    if (type=="func") {
        int NA = 4;
        float A[4] = {1., -1., 1., 1.};
        float* C = apply_fct(A, NA, "sigmoid");
        for (int i=0; i<NA; i++) {
            std::cout << C[i] << ' ';
        }
    }

    if (type=="softmax") {
        int NA = 4;
        float A[4] = {1., -1., 1., 1.};
        float* C = apply_softmax(A, NA);
        for (int i=0; i<NA; i++) {
            std::cout << C[i] << ' ';
        }
    }
}
