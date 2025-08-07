#include <torch/script.h>
#include <iostream>
#include <chrono>

// --- Parameters ---
const int NUM_ITERATIONS = 200;
const int WARMUP_ITERATIONS = 20;
const int BATCH_SIZE = 1;
const int NOISE_DIM = 100;

int main(int argc, const char* argv[]) {
    std::cout << "--- Performance Test: DCGAN with C++/LibTorch (TorchScript) ---" << std::endl;
    if (argc != 2) {
        std::cerr << "Usage: ./libtorch_infer <path_to_traced_model.pt>\n";
        return -1;
    }
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

    auto noise = torch::randn({BATCH_SIZE, NOISE_DIM, 1, 1}, device);
    
    std::cout << "Warming up...\n";
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            module.forward({noise});
        }
    }
    torch::cuda::synchronize();
    
    std::cout << "Starting benchmark...\n";
    auto start = std::chrono::steady_clock::now();
    {
        torch::NoGradGuard no_grad;
        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            module.forward({noise});
        }
    }
    torch::cuda::synchronize();
    auto end = std::chrono::steady_clock::now();

    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double avg_time_ms = static_cast<double>(total_time_ms) / NUM_ITERATIONS;

    std::cout << "\n--- LibTorch Results ---\n";
    std::cout << "Total time for " << NUM_ITERATIONS << " iterations: " << total_time_ms << " ms\n";
    printf("Average inference time: %.4f ms\n", avg_time_ms);

    return 0;
}