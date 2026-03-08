cmake_minimum_required(VERSION 4.1)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(spraw2 CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(spraw2 main.cu
        main.cu)

set_target_properties(spraw2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
