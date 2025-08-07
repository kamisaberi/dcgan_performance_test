# inference_pytorch.py
import torch
import torch.nn as nn
import time

# --- Parameters ---
NUM_ITERATIONS = 200
WARMUP_ITERATIONS = 20
BATCH_SIZE = 1
NOISE_DIM = 100

if __name__ == "__main__":
    print("--- Performance Test: DCGAN with Python/PyTorch ---")
    if not torch.cuda.is_available(): exit("CUDA not available.")
    
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Load pre-trained model
    model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=False)
    netG = model.netG.to(device)
    netG.eval()
    
    # Create fake input data
    noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1, device=device)
    
    print(f"Batch: {BATCH_SIZE}, Iterations: {NUM_ITERATIONS}")
    
    # Warm-up
    print("Warming up...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS):
            _ = netG(noise)
    torch.cuda.synchronize()
    
    # Benchmark
    print("Starting benchmark...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for i in range(NUM_ITERATIONS):
            _ = netG(noise)
    end_event.record()
    torch.cuda.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / NUM_ITERATIONS

    print("\n--- PyTorch Results ---")
    print(f"Total time for {NUM_ITERATIONS} iterations: {total_time_ms:.3f} ms")
    print(f"Average inference time: {avg_time_ms:.4f} ms")