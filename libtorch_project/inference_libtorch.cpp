#include <torch/script.h>
#include <iostream>
#include <chrono>
#include <vector>

// THE FIX: Add the required header for CUDA stream synchronization
#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
// --- Parameters ---
const int NUM_ITERATIONS = 200;
const int WARMUP_ITERATIONS = 20;
const int BATCH_SIZE = 1;
const int NOISE_DIM = 100;

int main(int argc, const char* argv[]) {
    std::cout << "--- Performance Test: DCGAN with C++/LibTorch (TorchScript) ---" << std::endl;
    if (argc != 2) {
        std::cerr << "Usage: ./libtorch_infer <path_to_model.pt>\n";
        return -1;
    }

    // THE FIX: Use C++ `::` syntax for namespaces
    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available.\n";
        return -1;
    }

    torch::Device device(torch::kCUDA);
    torch::jit::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return -1;
    }

    module.to(device);
    module.eval();

    // THE FIX: Use C++ `::` syntax for namespaces
    auto noise = torch::randn({BATCH_SIZE, NOISE_DIM, 1, 1}, device);

    // THE FIX: Use C++ `::` syntax for namespaces
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(noise);

    std::cout << "Warming up...\n";
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            module.forward(inputs);
        }
    }
    // THE FIX: This call now works because of the added header
    c10::cuda::getCurrentCUDAStream().synchronize();

    std::cout << "Starting benchmark...\n";
    auto start = std::chrono::high_resolution_clock::now();
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            module.forward(inputs);
        }
    }
    // THE FIX: This call now works because of the added header
    c10::cuda::getCurrentCUDAStream().synchronize();
    auto end = std::chrono::high_resolution_clock::now();

    auto total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double avg_time_ms = total_time_ms / NUM_ITERATIONS;

    std::cout << "\n--- LibTorch Results ---\n";
    printf("Total time for %d iterations: %.3f ms\n", (int)NUM_ITERATIONS, total_time_ms);
    printf("Average inference time: %.4f ms\n", avg_time_ms);

    return 0;
}