#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <curand.h>

// Helper Macros and Classes
#define CHECK_CUDA(call) { const cudaError_t e = call; if (e != cudaSuccess) { fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } }
#define CHECK_CURAND(call) { const curandStatus_t s = call; if (s != CURAND_STATUS_SUCCESS) { fprintf(stderr, "CURAND Error: %s:%d, %d\n", __FILE__, __LINE__, s); exit(1); } }
class GpuTimer { cudaEvent_t s, e; public: GpuTimer() { CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e)); } ~GpuTimer() { CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e)); } void start() { CHECK_CUDA(cudaEventRecord(s, 0)); } void stop() { CHECK_CUDA(cudaEventRecord(e, 0)); } float elapsed_ms() { float t; CHECK_CUDA(cudaEventSynchronize(e)); CHECK_CUDA(cudaEventElapsedTime(&t, s, e)); return t; } };

void load_weights(const std::string& filename, float* d_ptr, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { std::cerr << "FATAL: Could not open " << filename << std::endl; exit(1); }
    size_t size = file.tellg();
    if (size != num_elements * sizeof(float)) { std::cerr << "FATAL: Size mismatch for " << filename << std::endl; exit(1); }
    std::vector<float> h_weights(num_elements);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(h_weights.data()), size);
    CHECK_CUDA(cudaMemcpy(d_ptr, h_weights.data(), size, cudaMemcpyHostToDevice));
}

// Fused Kernels
__device__ inline float leaky_relu_device(float x) { return (x > 0.0f) ? x : 0.2f * x; }

__global__ void fused_deconv_bn_relu_kernel(const float* input, const float* weights, const float* bn_gamma, const float* bn_beta, const float* bn_mean, const float* bn_var, float* output, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int s, int p) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_c * out_h * out_w; idx += blockDim.x * gridDim.x) {
        int oc = idx / (out_h * out_w);
        int oy = (idx / out_w) % out_h;
        int ox = idx % out_w;
        float acc = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
            for (int r = 0; r < k; ++r) {
                for (int t = 0; t < k; ++t) {
                    int y_num = oy + p - r;
                    int x_num = ox + p - t;
                    if (y_num >= 0 && y_num % s == 0 && x_num >= 0 && x_num % s == 0) {
                        int iy = y_num / s;
                        int ix = x_num / s;
                        if (iy < in_h && ix < in_w) {
                            acc += input[ic * in_h * in_w + iy * in_w + ix] * weights[ic * out_c * k * k + oc * k * k + r * k + t];
                        }
                    }
                }
            }
        }
        const float epsilon = 1e-5f;
        float bn_out = bn_gamma[oc] * (acc - bn_mean[oc]) / sqrtf(bn_var[oc] + epsilon) + bn_beta[oc];
        output[idx] = leaky_relu_device(bn_out);
    }
}

__global__ void fused_deconv_tanh_kernel(const float* input, const float* weights, float* output, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int s, int p) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_c * out_h * out_w; idx += blockDim.x * gridDim.x) {
        int oc = idx / (out_h * out_w);
        int oy = (idx / out_w) % out_h;
        int ox = idx % out_w;
        float acc = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
            for (int r = 0; r < k; ++r) {
                for (int t = 0; t < k; ++t) {
                    int y_num = oy + p - r;
                    int x_num = ox + p - t;
                    if (y_num >= 0 && y_num % s == 0 && x_num >= 0 && x_num % s == 0) {
                        int iy = y_num / s;
                        int ix = x_num / s;
                        if (iy < in_h && ix < in_w) {
                            acc += input[ic * in_h * in_w + iy * in_w + ix] * weights[ic * out_c * k * k + oc * k * k + r * k + t];
                        }
                    }
                }
            }
        }
        output[idx] = tanhf(acc);
    }
}

// Main Program
const int NUM_ITERATIONS = 200;
const int WARMUP_ITERATIONS = 20;
const int BATCH_SIZE = 1;
const int NOISE_DIM = 100;

struct LayerParams { float *w, *gamma, *beta, *mean, *var; };

int main() {
    std::cout << "--- Performance Test: DCGAN with Custom CUDA ---" << std::endl;

    // Static Memory Plan
    const size_t buf1_size = 1024 * 4 * 4 * sizeof(float);
    const size_t buf2_size = 512 * 8 * 8 * sizeof(float);
    float *d_buf1, *d_buf2, *d_noise;
    CHECK_CUDA(cudaMalloc(&d_noise, BATCH_SIZE * NOISE_DIM * 1 * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_buf1, std::max(buf1_size, 256 * 16 * 16 * sizeof(float))));
    CHECK_CUDA(cudaMalloc(&d_buf2, std::max(buf2_size, 128 * 32 * 32 * sizeof(float))));

    // Final output buffer
    float *d_final_output;
    CHECK_CUDA(cudaMalloc(&d_final_output, BATCH_SIZE * 3 * 64 * 64 * sizeof(float)));

    // Allocate and load weights
    LayerParams l1, l2, l3, l4;
    float* l5_w;
    CHECK_CUDA(cudaMalloc(&l1.w, 100 * 1024 * 4 * 4 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l1.gamma, 1024 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l1.beta, 1024 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l1.mean, 1024 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l1.var, 1024 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l2.w, 1024 * 512 * 4 * 4 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l2.gamma, 512 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l2.beta, 512 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l2.mean, 512 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l2.var, 512 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l3.w, 512 * 256 * 4 * 4 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l3.gamma, 256 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l3.beta, 256 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l3.mean, 256 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l3.var, 256 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l4.w, 256 * 128 * 4 * 4 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l4.gamma, 128 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l4.beta, 128 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l4.mean, 128 * sizeof(float))); CHECK_CUDA(cudaMalloc(&l4.var, 128 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l5_w, 128 * 3 * 4 * 4 * sizeof(float)));

    load_weights("../../weights/layer1_weight.bin", l1.w, 100*1024*16); load_weights("../../weights/layer1_bn_gamma.bin", l1.gamma, 1024); load_weights("../../weights/layer1_bn_beta.bin", l1.beta, 1024); load_weights("../../weights/layer1_bn_mean.bin", l1.mean, 1024); load_weights("../../weights/layer1_bn_var.bin", l1.var, 1024);
    load_weights("../../weights/layer2_weight.bin", l2.w, 1024*512*16); load_weights("../../weights/layer2_bn_gamma.bin", l2.gamma, 512); load_weights("../../weights/layer2_bn_beta.bin", l2.beta, 512); load_weights("../../weights/layer2_bn_mean.bin", l2.mean, 512); load_weights("../../weights/layer2_bn_var.bin", l2.var, 512);
    load_weights("../../weights/layer3_weight.bin", l3.w, 512*256*16); load_weights("../../weights/layer3_bn_gamma.bin", l3.gamma, 256); load_weights("../../weights/layer3_bn_beta.bin", l3.beta, 256); load_weights("../../weights/layer3_bn_mean.bin", l3.mean, 256); load_weights("../../weights/layer3_bn_var.bin", l3.var, 256);
    load_weights("../../weights/layer4_weight.bin", l4.w, 256*128*16); load_weights("../../weights/layer4_bn_gamma.bin", l4.gamma, 128); load_weights("../../weights/layer4_bn_beta.bin", l4.beta, 128); load_weights("../../weights/layer4_bn_mean.bin", l4.mean, 128); load_weights("../../weights/layer4_bn_var.bin", l4.var, 128);
    load_weights("../../weights/layer5_weight.bin", l5_w, 128*3*16);

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    std::cout << "Warming up...\n";
    for(int i=0; i<WARMUP_ITERATIONS; ++i) {
        CHECK_CURAND(curandGenerateNormal(gen, d_noise, BATCH_SIZE * NOISE_DIM, 0.0f, 1.0f));
        fused_deconv_bn_relu_kernel<<<512, 256>>>(d_noise, l1.w, l1.gamma, l1.beta, l1.mean, l1.var, d_buf1, 100, 1, 1, 1024, 4, 4, 4, 1, 0);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Starting benchmark...\n";
    GpuTimer timer;
    float total_time = 0;
    dim3 grid(512); dim3 block(256);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        CHECK_CURAND(curandGenerateNormal(gen, d_noise, BATCH_SIZE * NOISE_DIM, 0.0f, 1.0f));
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_noise, l1.w, l1.gamma, l1.beta, l1.mean, l1.var, d_buf1, 100, 1, 1, 1024, 4, 4, 4, 1, 0);
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_buf1, l2.w, l2.gamma, l2.beta, l2.mean, l2.var, d_buf2, 1024, 4, 4, 512, 8, 8, 4, 2, 1);
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_buf2, l3.w, l3.gamma, l3.beta, l3.mean, l3.var, d_buf1, 512, 8, 8, 256, 16, 16, 4, 2, 1);
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_buf1, l4.w, l4.gamma, l4.beta, l4.mean, l4.var, d_buf2, 256, 16, 16, 128, 32, 32, 4, 2, 1);
        fused_deconv_tanh_kernel<<<grid, block>>>(d_buf2, l5_w, d_final_output, 128, 32, 32, 3, 64, 64, 4, 2, 1);
        timer.stop();
        total_time += timer.elapsed_ms();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "\n--- Custom CUDA Results ---\n";
    printf("Total time for %d iterations: %.3f ms\n", NUM_ITERATIONS, total_time);
    printf("Average inference time: %.4f ms\n", total_time / NUM_ITERATIONS);

    // ... Cleanup would go here ...
    return 0;
}