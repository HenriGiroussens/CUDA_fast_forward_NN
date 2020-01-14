//
// Created by henri on 14/01/2020.
//

#include "load_weights.hh"

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>


void split(const std::string& txt, double* strs)
{
    size_t pos = txt.find(' ');
    size_t initialPos = 0;
    int wheretoinsert = 0;
    while (pos != std::string::npos) {
        strs[wheretoinsert] = stod(txt.substr(initialPos, pos - initialPos));
        initialPos = pos + 1;
        wheretoinsert = wheretoinsert + 1;
        pos = txt.find(' ', initialPos);
    }
    if (wheretoinsert < 9) {
        strs[wheretoinsert] = stod(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));
    }

    return;
}


void compute(std::string inputshape, int* res) {
    for (int i = 0; i < 4 ; ++i) {
        res[i] = 0;
    }
    int casetofill = 0;
    int temp = 0;
    for (int i = 0; i < inputshape.length(); ++i) {
        if (inputshape[i] == '(') {
            continue;
        }
        else if (inputshape[i] == ',') {
            casetofill = casetofill + 1;
            temp = 0;
        }
        else if (inputshape[i] == ')') {
            return;
        }
        else if (inputshape[i] > 47 && inputshape[i] < 58) {
            res[casetofill] = res[casetofill] * 10 + inputshape[i] - 48;

        }

    }

}
int main() {

	auto path = "../data/weights.txt";

    std::ifstream input(path);
    for (std::string line; getline(input, line); )
    {
        if (line == "conv2D - weight") {
            std::cout << line << std::endl;
            std::string shape;
            getline(input, shape);
            int shapearray[4];
            compute(shape, shapearray);
            std::cout << shape << std::endl;
            std::cout << "shapearray : " << shapearray[0] << " " << shapearray[1] << " " << shapearray[2] << " " << shapearray[3] << std::endl;
            double*** layer = new double**[shapearray[0]];
            for (int i = 0; i < shapearray[0]; ++i) {
                double** neural = new double*[shapearray[1]];
                for (int j = 0; j < shapearray[1]; ++j) {
                    double* weights = new double[shapearray[2] * shapearray[3]];
                    std::string weightsline;
                    getline(input, weightsline);
                    split(weightsline, weights);
                    neural[j] = weights;

                }
                layer[i] = neural;
            }

        }
        else if (line == "conv2D - bias") {
            std::cout << line << std::endl;

            std::string shape;
            getline(input, shape);
            int shapearray[4];
            compute(shape, shapearray);
            std::cout << shape << std::endl;
            std::cout << "shapearray : " << shapearray[0]  << std::endl;

            double* weights = new double[shapearray[0]];
            std::string weightsline;
            getline(input, weightsline);
            split(weightsline, weights);

        }
        else if (line == "Dense - weight") {
            std::cout << line << std::endl;

            std::string shape;
            getline(input, shape);
            int shapearray[4];
            compute(shape, shapearray);
            std::cout << shape << std::endl;
            std::cout << "shapearray : " << shapearray[0] << " " << shapearray[1]  << std::endl;
            double* weights = new double[shapearray[0]];

            double** layer = new double* [shapearray[0]];
            for (int i = 0; i < shapearray[0]; ++i) {
                double* weights = new double[shapearray[1]];
                std::string weightsline;
                getline(input, weightsline);
                if (shapearray[1] != 1)
                    split(weightsline, weights);
                else
                    weights[0] = stod(weightsline);
                layer[i] = weights;

            }
        }
        else if (line == "Dense - bias") {
            std::cout << line << std::endl;

            std::string shape;
            getline(input, shape);
            int shapearray[4];
            compute(shape, shapearray);
            std::cout << shape << std::endl;
            std::cout << "shapearray : " << shapearray[0] << std::endl;

            double* weights = new double[shapearray[0]];
            std::string weightsline;
            getline(input, weightsline);
            if (shapearray[0] != 1)
                split(weightsline, weights);
            else
                weights[0] = stod(weightsline);
        }
    }
}