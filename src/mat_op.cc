#include "matrices_operations/matrix_add.hh"
#include "matrices_operations/matrix_mult.hh"
#include "matrices_operations/matrix_conv.hh"
#include "matrices_operations/apply_fct.hh"
#include "matrices_operations/apply_softmax.hh"
#include "matrices_operations/matrix_avg_pooling.hh"


int main() {
    std::string type("pooling");

    if (type=="add") {
        int NA = 4;
        int MA = 5;
        int NB = 4;
        int MB = 5;
        double A[20] =
                {3., 3., 2., 1., 0.,
                0., 0., 1., 3., 1.,
                3., 1., 2., 2., 3.,
                2., 0., 0., 2., 2.};
        double B[20] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};
        double* C = mat_add(A, B, NA, MA, NB, MB);
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
        double A[6] =
                {3., 3., 2.,
                 0., 0., 1.};
        double B[12] =
                {1., 1., 1., 1.,
                 1., 1., 1., 1.,
                 1., 1., 1., 1.};
        double* C = mat_mult(A, B, NA, MA, NB, MB);
        for (int i=0; i<NA; i++) {
            for (int j = 0; j < MB; j++) {
                std::cout << C[i * MB + j] << ' ';
            }
            std::cout << '\n';
        }
    }


    if (type=="test_dense") {
        double A[9] =
                {12., 12., 17., 10., 17., 19., 9., 6., 14.};
        double W2[18] =
                {1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.,
                 -1., -1.,
                 1., 1.};
        double B2[2] = {1., 1.};
        double* C = mat_mult(A, W2, 1, 9, 9, 2);
        double* res = mat_add(C, B2, 1, 2, 1, 2);
        for (int i=0; i<1; i++) {
            for (int j = 0; j < 2; j++) {
                std::cout << res[i * 2 + j] << ' ';
            }
            std::cout << '\n';
        }
    }



    if (type=="conv_valid") {
        int NA = 4;
        int MA = 5;
        int KN = 3;
        double A[20] =
                {3., 3., 2., 1., 0.,
                 0., 0., 1., 3., 1.,
                 3., 1., 2., 2., 3.,
                 2., 0., 0., 2., 2.};

        double K[9] = {0., 1., 2.,
                      2., 2., 0.,
                      0., 1., 2.};
        double* C = mat_conv(A, K, NA, MA, KN, "valid");
        for (int i=0; i<NA - 2*(KN/2); i++) {
            for (int j = 0; j < MA - 2*(KN/2); j++) {
                std::cout << C[i * (MA - 2*(KN/2)) + j] << ' ';
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }


    if (type=="conv_same") {
        int NA = 4;
        int MA = 5;
        int KN = 3;
        double A[20] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};

        double K[9] = {1., 10., 100.,
                      1000., 10000., 100000.,
                      1000000., 10000000., 100000000.};
        double* C = mat_conv(A, K, NA, MA, KN, "same");
        for (int i=0; i<NA; i++) {
            for (int j = 0; j < MA; j++) {
                std::cout << C[i * MA + j] << ' ';
            }
            std::cout << '\n';
        }
    }


    if (type=="func") {
        int NA = 4;
        double A[4] = {1., -1., 1., 1.};
        double* C = apply_fct(A, NA, "exp");
        for (int i=0; i<NA; i++) {
            std::cout << C[i] << ' ';
        }
    }

    if (type=="softmax") {
        int NA = 4;
        double A[4] = {1., -1., 1., 1.};
        double* C = apply_softmax(A, NA);
        for (int i=0; i<NA; i++) {
            std::cout << C[i] << ' ';
        }
    }

    if (type=="pooling") {
        int NA = 4;
        int MA = 5;
        double A[20] =
                {1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.,
                 1., 1., 1., 1., 1.};

        double* C = avg_pooling_2D(A, 4, 5, 2, "valid");
        for (int i=0; i<2; i++) {
            for (int j = 0; j < 2; j++) {
                std::cout << C[i * 2 + j] << ' ';
            }
            std::cout << '\n';
        }
    }
}
