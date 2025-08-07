 
cmake_minimum_required(VERSION 3.18)
project(LibTorchInference)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

add_executable(libtorch_infer inference_libtorch.cpp)

target_link_libraries(libtorch_infer "${TORCH_LIBRARIES}")
set_property(TARGET libtorch_infer PROPERTY CXX_STANDARD 17)