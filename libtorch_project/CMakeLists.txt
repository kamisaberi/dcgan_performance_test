 
cmake_minimum_required(VERSION 3.18)
project(LibTorchInference)

set(CMAKE_C_COMPILER "gcc-12")
set(CMAKE_CXX_COMPILER "g++-12")
set(CMAKE_CUDA_HOST_COMPILER "g++-12")
set(CMAKE_CUDA_ARCHITECTURES "75;86")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

list(APPEND CMAKE_PREFIX_PATH /home/kami/libs/cpp/libtorch/)


find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

add_executable(libtorch_infer inference_libtorch.cpp)

target_link_libraries(libtorch_infer "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_infer PROPERTY CXX_STANDARD 17)