#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <numeric>
#include <cuda_runtime_api.h>
#include <curand.h>

#include "NvInfer.h"
#include "NvOnnxParser.h"

// ===================================================================
//                        HELPER CODE
// ===================================================================

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CURAND(call) { \
    const curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "CURAND Error: %s:%d, %d\n", __FILE__, __LINE__, status); \
        exit(1); \
    } \
}

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

class GpuTimer {
    cudaEvent_t start_event, stop_event;
public:
    GpuTimer() { CHECK_CUDA(cudaEventCreate(&start_event)); CHECK_CUDA(cudaEventCreate(&stop_event)); }
    ~GpuTimer() { CHECK_CUDA(cudaEventDestroy(start_event)); CHECK_CUDA(cudaEventDestroy(stop_event)); }
    void start() { CHECK_CUDA(cudaEventRecord(start_event, 0)); }
    void stop() { CHECK_CUDA(cudaEventRecord(stop_event, 0)); }
    float elapsed_ms() { float t; CHECK_CUDA(cudaEventSynchronize(stop_event)); CHECK_CUDA(cudaEventElapsedTime(&t, start_event, stop_event)); return t; }
};


// ===================================================================
//                          MAIN PROGRAM
// ===================================================================
const int NUM_ITERATIONS = 200;
const int WARMUP_ITERATIONS = 20;
const int BATCH_SIZE = 1;
const int NOISE_DIM = 100;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./tensorrt_infer <path_to_onnx_model.onnx>" << std::endl;
        return -1;
    }
    const std::string onnx_file_path = argv[1];
    const std::string engine_file_path = "generator.engine";

    Logger logger;

    // --- Build or Load TensorRT Engine ---
    nvinfer1::ICudaEngine* engine = nullptr;
    std::ifstream engine_file(engine_file_path, std::ios::binary);
    if (engine_file) {
        std::cout << "Loading existing TensorRT engine from " << engine_file_path << std::endl;
        engine_file.seekg(0, std::ios::end);
        size_t engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> engine_buffer(engine_size);
        engine_file.read(engine_buffer.data(), engine_size);

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(engine_buffer.data(), engine_size);
        runtime->destroy();
    } else {
        std::cout << "Building TensorRT engine... (This will take a few minutes the first time)" << std::endl;
        nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

        parser->parseFromFile(onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

        nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
        if (builder->platformHasFastFp16()) {
            std::cout << "Enabling FP16 mode." << std::endl;
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        nvinfer1::IHostMemory* serialized_engine = builder->buildSerializedNetwork(*network, *config);

        std::ofstream out_engine_file(engine_file_path, std::ios::binary);
        out_engine_file.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size());

        serialized_engine->destroy();
        config->destroy();
        parser->destroy();
        network->destroy();
        builder->destroy();
        runtime->destroy();
    }

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // --- Allocate Buffers ---
    void* buffers[2];
    const int input_idx = engine->getBindingIndex("input");
    const int output_idx = engine->getBindingIndex("output");

    size_t input_size = (size_t)BATCH_SIZE * NOISE_DIM * 1 * 1 * sizeof(float);
    size_t output_size = (size_t)BATCH_SIZE * 3 * 64 * 64 * sizeof(float);

    CHECK_CUDA(cudaMalloc(&buffers[input_idx], input_size));
    CHECK_CUDA(cudaMalloc(&buffers[output_idx], output_size));

    // --- Create cuRAND generator for noise ---
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    std::cout << "\nWarming up...\n";
    for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
        CHECK_CURAND(curandGenerateNormal(gen, (float*)buffers[input_idx], BATCH_SIZE * NOISE_DIM, 0.0f, 1.0f));
        context->enqueueV2(buffers, 0, nullptr);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Starting benchmark...\n";
    GpuTimer timer;
    float total_time = 0;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        CHECK_CURAND(curandGenerateNormal(gen, (float*)buffers[input_idx], BATCH_SIZE * NOISE_DIM, 0.0f, 1.0f));
        context->enqueueV2(buffers, 0, nullptr);
        timer.stop();
        total_time += timer.elapsed_ms();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "\n--- TensorRT C++ Results ---\n";
    printf("Total time for %d iterations: %.3f ms\n", NUM_ITERATIONS, total_time);
    printf("Average inference time: %.4f ms\n", total_time / NUM_ITERATIONS);

    // Cleanup
    CHECK_CUDA(cudaFree(buffers[input_idx]));
    CHECK_CUDA(cudaFree(buffers[output_idx]));
    context->destroy();
    engine->destroy();
    curandDestroyGenerator(gen);

    return 0;
}