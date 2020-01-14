//
// Created by henri on 14/01/2020.
//

#include "load_weights.hh"



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
Model* get_model() {

	auto path = "../data/weights.txt";


    auto* conv_1_1 = new Conv2D(48, 48, 1, 32, 3, true);
    auto* act_1_1 = new Activation("relu", 46*46, 32);
    auto* conv_1_2 = new Conv2D(46, 46, 32, 32, 3, true);
    auto* act_1_2 = new Activation("relu", 44*44, 32);
    auto* conv_1_3 = new Conv2D(44, 44, 32, 32, 3, true);
    auto* act_1_3 = new Activation("relu", 42*42, 32);
    auto* avg_1 = new AveragePooling2D(42, 42, 32, 2, true);


    auto* conv_2_1 = new Conv2D(21, 21, 32,64, 3, true);
    auto* act_2_1 = new Activation("relu", 19*19, 64);
    auto* conv_2_2 = new Conv2D(19, 19, 64, 64, 3, true);
    auto* act_2_2 = new Activation("relu", 17*17, 64);
    auto* conv_2_3 = new Conv2D(17, 17, 64, 64, 3, true);
    auto* act_2_3 = new Activation("relu", 15*15, 64);
    auto* avg_2 = new AveragePooling2D(16, 16, 64, 5, true);

    auto* flatten = new Flatten(3*3, 64);
    auto* dense_1 = new Dense(3*3*64, 64);
    auto* act_1 = new Activation("relu", 64, 1);
    auto* dense_2 = new Dense(64, 32);
    auto* act_2 = new Activation("relu", 32, 1);
    auto dense_3 = new Dense(32, 1);
    auto* act_3 = new Activation("sigmoid", 1, 1);

    Conv2D *conv_layers[6] = {conv_1_1, conv_1_2, conv_1_3, conv_2_1, conv_2_2, conv_2_3};
    Dense *dense_layers[3] = {dense_1, dense_2, dense_3};
    std::vector<Layer*> all_layers ({conv_1_1, act_1_1, conv_1_2, act_1_2, conv_1_3, act_1_3, avg_1,
                             conv_2_1, act_2_1, conv_2_2, act_2_2, conv_2_3, act_2_3, avg_2,
                             flatten, dense_1, act_1, dense_2, act_2, dense_3, act_3});

    std::ifstream input(path);

	std::vector<Layer> layers;
	int current_layer = 0;
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
            double*** kernels = new double**[shapearray[0]];
            for (int i = 0; i < shapearray[0]; ++i) {
                double** neural = new double*[shapearray[1]];
                for (int j = 0; j < shapearray[1]; ++j) {
                    double* weights = new double[shapearray[2] * shapearray[3]];
                    std::string weightsline;
                    getline(input, weightsline);
                    split(weightsline, weights);
                    neural[j] = weights;

                }
                kernels[i] = neural;
            }
            conv_layers[current_layer]->setKernels(kernels);
        }
        else if (line == "conv2D - bias") {
            std::cout << line << std::endl;

            std::string shape;
            getline(input, shape);
            int shapearray[4];
            compute(shape, shapearray);
            std::cout << shape << std::endl;
            std::cout << "shapearray : " << shapearray[0]  << std::endl;

            double* bias = new double[shapearray[0]];
            std::string weightsline;
            getline(input, weightsline);
            split(weightsline, bias);
            conv_layers[current_layer]->setBiasVector(bias);
            current_layer++;
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
            dense_layers[current_layer - 6]->setWeightMatrix(weights);
        }
        else if (line == "Dense - bias") {
            std::cout << line << std::endl;

            std::string shape;
            getline(input, shape);
            int shapearray[4];
            compute(shape, shapearray);
            std::cout << shape << std::endl;
            std::cout << "shapearray : " << shapearray[0] << std::endl;

            double* bias = new double[shapearray[0]];
            std::string weightsline;
            getline(input, weightsline);
            if (shapearray[0] != 1)
                split(weightsline, bias);
            else
                bias[0] = stod(weightsline);
            dense_layers[current_layer - 6]->setBiasVector(bias);
            current_layer++;
        }
    }
    Model* model = new Model(all_layers, 18, 0, 0);
    return model;
}
