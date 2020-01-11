cmake_minimum_required (VERSION 3.14)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
project (GPGPU CUDA CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )
find_package(PNG REQUIRED)
find_package(CUDA REQUIRED)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include(FetchContent)
FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.4.2.tar.gz
  )



add_executable(mat_op
        src/matrices_operations/matrix_add.cu
        src/matrices_operations/matrix_mult.cu
        src/matrices_operations/kernels/kernel_mat_op.cu
        src/matrices_operations/matrix_conv.cu
        src/matrices_operations/mat_op.cc
        src/matrices_operations/apply_fct.cc
        src/matrices_operations/apply_softmax.cc)
target_compile_features(mat_op PUBLIC)
target_include_directories(mat_op PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(mat_op PRIVATE ${CUDART_LIBRARY})


add_executable(network
        src/matrices_operations/matrix_add.cu
        src/matrices_operations/matrix_mult.cu
        src/matrices_operations/kernels/kernel_mat_op.cu
        src/matrices_operations/matrix_conv.cu
        src/network.cc
        src/layers/Activation.cc
        src/matrices_operations/apply_fct.cc
        src/matrices_operations/apply_softmax.cc)
target_compile_features(network PUBLIC)
target_include_directories(network PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(network PRIVATE ${CUDART_LIBRARY})



