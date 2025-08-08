cmake_minimum_required(VERSION 3.18)
project(TensorRTInference LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA REQUIRED)

# Find TensorRT
# You might need to adjust this path to where your TensorRT is installed
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/x86_64-linux-gnu/
)
find_library(TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/x86_64-linux-gnu/
)
find_library(TENSORRT_PARSER_LIBRARY nvinfer_plugin
        HINTS /usr/lib/x86_64-linux-gnu/
)
find_library(TENSORRT_ONNX_PARSER_LIBRARY nvonnxparser
        HINTS /usr/lib/x86_64-linux-gnu/
)

if (NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please set TENSORRT_ROOT or check paths.")
endif()

message(STATUS "Found TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "Found TensorRT library: ${TENSORRT_LIBRARY}")

add_executable(tensorrt_infer inference_tensorrt.cpp)

# Link all necessary libraries
target_include_directories(tensorrt_infer PRIVATE ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})
target_link_libraries(tensorrt_infer PRIVATE
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY}
        ${TENSORRT_LIBRARY}
        ${TENSORRT_PARSER_LIBRARY}
        ${TENSORRT_ONNX_PARSER_LIBRARY}
)

set_property(TARGET tensorrt_infer PROPERTY CXX_STANDARD 17)