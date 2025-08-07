 
cmake_minimum_required(VERSION 3.18)
project(CustomCUDAInference LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

add_executable(custom_cuda_infer inference_custom_cuda.cu)

target_link_libraries(custom_cuda_infer PRIVATE ${CUDA_curand_LIBRARY})

set_target_properties(custom_cuda_infer PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)