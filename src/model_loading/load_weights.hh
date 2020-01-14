//
// Created by henri on 14/01/2020.
//

#ifndef GPGPU_LOAD_WEIGHTS_HH
#define GPGPU_LOAD_WEIGHTS_HH

#include "../layers/Layer.hh"
#include "../layers/Model.hh"
#include "../layers/Conv2D.hh"
#include "../layers/AveragePooling2D.hh"
#include "../layers/Flatten.hh"
#include "../layers/Dense.hh"
#include "../layers/Activation.hh"

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <string>


Model* get_model();

#endif //GPGPU_LOAD_WEIGHTS_HH
