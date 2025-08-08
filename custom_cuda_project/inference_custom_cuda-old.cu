#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <curand.h>

// ===================================================================
//                        CORRECTED HELPER CODE
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

// THE FIX: This is the full, correct, multi-line GpuTimer class definition.
// The previous one-liner was buggy and caused the compiler errors.
class GpuTimer {
    cudaEvent_t start_event, stop_event;
public:
    GpuTimer() {
        CHECK_CUDA(cudaEventCreate(&start_event));
        CHECK_CUDA(cudaEventCreate(&stop_event));
    }
    ~GpuTimer() {
        CHECK_CUDA(cudaEventDestroy(start_event));
        CHECK_CUDA(cudaEventDestroy(stop_event));
    }
    void start() {
        CHECK_CUDA(cudaEventRecord(start_event, 0));
    }
    void stop() {
        CHECK_CUDA(cudaEventRecord(stop_event, 0));
    }
    float elapsed_ms() {
        float time_ms;
        CHECK_CUDA(cudaEventSynchronize(stop_event));
        CHECK_CUDA(cudaEventElapsedTime(&time_ms, start_event, stop_event));
        return time_ms;
    }
};

void load_weights(const std::string& filename, float* d_ptr, size_t num_elements);

// ===================================================================
//                        KERNELS (Unchanged)
// ===================================================================
__device__ inline float relu_device(float x) { return fmaxf(0.0f, x); }

__global__ void fused_deconv_bn_relu_kernel(const float* input, const float* weights, const float* gamma, const float* beta, const float* mean, const float* var, float* output, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int s, int p) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_c * out_h * out_w; idx += blockDim.x * gridDim.x) {
        int oc = idx / (out_h * out_w); int oy = (idx / out_w) % out_h; int ox = idx % out_w;
        float acc = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
            for (int r = 0; r < k; ++r) {
                for (int t = 0; t < k; ++t) {
                    int y_num = oy + p - r; int x_num = ox + p - t;
                    if (y_num >= 0 && y_num % s == 0 && x_num >= 0 && x_num % s == 0) {
                        int iy = y_num / s; int ix = x_num / s;
                        if (iy < in_h && ix < in_w) {
                            acc += input[ic * in_h * in_w + iy * in_w + ix] * weights[ic * out_c * k * k + oc * k * k + r * k + t];
                        }
                    }
                }
            }
        }
        const float epsilon = 1e-5f;
        float bn_out = gamma[oc] * (acc - mean[oc]) / sqrtf(var[oc] + epsilon) + beta[oc];
        output[idx] = relu_device(bn_out);
    }
}

__global__ void fused_deconv_tanh_kernel(const float* input, const float* weights, float* output, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int s, int p) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < out_c * out_h * out_w; idx += blockDim.x * gridDim.x) {
        int oc = idx / (out_h * out_w); int oy = (idx / out_w) % out_h; int ox = idx % out_w;
        float acc = 0.0f;
        for (int ic = 0; ic < in_c; ++ic) {
            for (int r = 0; r < k; ++r) {
                for (int t = 0; t < k; ++t) {
                    int y_num = oy + p - r; int x_num = ox + p - t;
                    if (y_num >= 0 && y_num % s == 0 && x_num >= 0 && x_num % s == 0) {
                        int iy = y_num / s; int ix = x_num / s;
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

// ===================================================================
//                        MAIN PROGRAM
// ===================================================================
const int NUM_ITERATIONS = 200;
const int WARMUP_ITERATIONS = 20;
const int BATCH_SIZE = 1;
const int NOISE_DIM = 100;
const int NGF = 128; // Number of generator features, matching Python script

struct LayerParams { float *w; };
struct BNParams { float *gamma, *beta, *mean, *var; };

int main() {
    std::cout << "--- Performance Test: DCGAN with Custom CUDA ---" << std::endl;
    float *d_buf1, *d_buf2, *d_noise;
    size_t max_buffer_elements = (size_t)NGF * 8 * 4 * 4;
    CHECK_CUDA(cudaMalloc(&d_noise, (size_t)BATCH_SIZE * NOISE_DIM * 1 * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_buf1, max_buffer_elements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_buf2, max_buffer_elements * sizeof(float)));
    float *d_ping = d_buf1; float *d_pong = d_buf2;
    float *d_final_output; CHECK_CUDA(cudaMalloc(&d_final_output, (size_t)BATCH_SIZE * 3 * 64 * 64 * sizeof(float)));

    LayerParams l1, l2, l3, l4, l5; BNParams bn1, bn2, bn3, bn4;
    CHECK_CUDA(cudaMalloc(&l1.w, (size_t)NOISE_DIM*NGF*8*16*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn1.gamma, (size_t)NGF*8*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn1.beta, (size_t)NGF*8*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn1.mean, (size_t)NGF*8*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn1.var, (size_t)NGF*8*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l2.w, (size_t)NGF*8*NGF*4*16*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn2.gamma, (size_t)NGF*4*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn2.beta, (size_t)NGF*4*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn2.mean, (size_t)NGF*4*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn2.var, (size_t)NGF*4*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l3.w, (size_t)NGF*4*NGF*2*16*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn3.gamma, (size_t)NGF*2*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn3.beta, (size_t)NGF*2*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn3.mean, (size_t)NGF*2*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn3.var, (size_t)NGF*2*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l4.w, (size_t)NGF*2*NGF*1*16*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn4.gamma, (size_t)NGF*1*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn4.beta, (size_t)NGF*1*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn4.mean, (size_t)NGF*1*sizeof(float))); CHECK_CUDA(cudaMalloc(&bn4.var, (size_t)NGF*1*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l5.w, (size_t)NGF*1*3*16*sizeof(float)));
    load_weights("../../weights/layer1_weight.bin", l1.w, (size_t)NOISE_DIM*NGF*8*16); load_weights("../../weights/layer1_bn_gamma.bin", bn1.gamma, (size_t)NGF*8); load_weights("../../weights/layer1_bn_beta.bin", bn1.beta, (size_t)NGF*8); load_weights("../../weights/layer1_bn_mean.bin", bn1.mean, (size_t)NGF*8); load_weights("../../weights/layer1_bn_var.bin", bn1.var, (size_t)NGF*8);
    load_weights("../../weights/layer2_weight.bin", l2.w, (size_t)NGF*8*NGF*4*16); load_weights("../../weights/layer2_bn_gamma.bin", bn2.gamma, (size_t)NGF*4); load_weights("../../weights/layer2_bn_beta.bin", bn2.beta, (size_t)NGF*4); load_weights("../../weights/layer2_bn_mean.bin", bn2.mean, (size_t)NGF*4); load_weights("../../weights/layer2_bn_var.bin", bn2.var, (size_t)NGF*4);
    load_weights("../../weights/layer3_weight.bin", l3.w, (size_t)NGF*4*NGF*2*16); load_weights("../../weights/layer3_bn_gamma.bin", bn3.gamma, (size_t)NGF*2); load_weights("../../weights/layer3_bn_beta.bin", bn3.beta, (size_t)NGF*2); load_weights("../../weights/layer3_bn_mean.bin", bn3.mean, (size_t)NGF*2); load_weights("../../weights/layer3_bn_var.bin", bn3.var, (size_t)NGF*2);
    load_weights("../../weights/layer4_weight.bin", l4.w, (size_t)NGF*2*NGF*1*16); load_weights("../../weights/layer4_bn_gamma.bin", bn4.gamma, (size_t)NGF*1); load_weights("../../weights/layer4_bn_beta.bin", bn4.beta, (size_t)NGF*1); load_weights("../../weights/layer4_bn_mean.bin", bn4.mean, (size_t)NGF*1); load_weights("../../weights/layer4_bn_var.bin", bn4.var, (size_t)NGF*1);
    load_weights("../../weights/layer5_weight.bin", l5.w, (size_t)NGF*1*3*16);

    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

    std::cout << "Warming up...\n";
    for(int i=0; i<WARMUP_ITERATIONS; ++i) { CHECK_CURAND(curandGenerateNormal(gen, d_noise, (size_t)BATCH_SIZE*NOISE_DIM, 0.0f, 1.0f)); fused_deconv_bn_relu_kernel<<<512, 256>>>(d_noise, l1.w, bn1.gamma, bn1.beta, bn1.mean, bn1.var, d_ping, NOISE_DIM, 1, 1, NGF*8, 4, 4, 4, 1, 0); }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "Starting benchmark...\n";
    GpuTimer timer; float total_time = 0; dim3 grid(1024); dim3 block(256);

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        timer.start();
        CHECK_CURAND(curandGenerateNormal(gen, d_noise, (size_t)BATCH_SIZE*NOISE_DIM, 0.0f, 1.0f));
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_noise, l1.w, bn1.gamma, bn1.beta, bn1.mean, bn1.var, d_ping, NOISE_DIM, 1, 1, NGF*8, 4, 4, 4, 1, 0);
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_ping, l2.w, bn2.gamma, bn2.beta, bn2.mean, bn2.var, d_pong, NGF*8, 4, 4, NGF*4, 8, 8, 4, 2, 1);
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_pong, l3.w, bn3.gamma, bn3.beta, bn3.mean, bn3.var, d_ping, NGF*4, 8, 8, NGF*2, 16, 16, 4, 2, 1);
        fused_deconv_bn_relu_kernel<<<grid, block>>>(d_ping, l4.w, bn4.gamma, bn4.beta, bn4.mean, bn4.var, d_pong, NGF*2, 16, 16, NGF, 32, 32, 4, 2, 1);
        fused_deconv_tanh_kernel<<<grid, block>>>(d_pong, l5.w, d_final_output, NGF, 32, 32, 3, 64, 64, 4, 2, 1);
        timer.stop();
        total_time += timer.elapsed_ms();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    std::cout << "\n--- Custom CUDA Results ---\n";
    printf("Total time for %d iterations: %.3f ms\n", NUM_ITERATIONS, total_time);
    printf("Average inference time: %.4f ms\n", total_time / NUM_ITERATIONS);

    // Cleanup
    cudaFree(d_noise); cudaFree(d_buf1); cudaFree(d_buf2); cudaFree(d_final_output);
    cudaFree(l1.w); cudaFree(bn1.gamma); cudaFree(bn1.beta); cudaFree(bn1.mean); cudaFree(bn1.var);
    cudaFree(l2.w); cudaFree(bn2.gamma); cudaFree(bn2.beta); cudaFree(bn2.mean); cudaFree(bn2.var);
    cudaFree(l3.w); cudaFree(bn3.gamma); cudaFree(bn3.beta); cudaFree(bn3.mean); cudaFree(bn3.var);
    cudaFree(l4.w); cudaFree(bn4.gamma); cudaFree(bn4.beta); cudaFree(bn4.mean); cudaFree(bn4.var);
    cudaFree(l5.w);
    curandDestroyGenerator(gen);

    return 0;
}

void load_weights(const std::string& filename, float* d_ptr, size_t num_elements) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) { std::cerr << "FATAL: Could not open " << filename << std::endl; exit(1); }
    size_t size = file.tellg();
    if (size != num_elements * sizeof(float)) { std::cerr << "FATAL: Size mismatch for " << filename << ". Expected " << num_elements * sizeof(float) << ", got " << size << std::endl; exit(1); }
    std::vector<float> h_weights(num_elements);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(h_weights.data()), size);
    CHECK_CUDA(cudaMemcpy(d_ptr, h_weights.data(), size, cudaMemcpyHostToDevice));
}