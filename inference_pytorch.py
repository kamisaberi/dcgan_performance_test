import torch
import time

NUM_ITERATIONS = 200
WARMUP_ITERATIONS = 20
BATCH_SIZE = 1
NOISE_DIM = 100

if __name__ == "__main__":
    print("--- Performance Test: DCGAN with Python/PyTorch (Traced Model) ---")
    if not torch.cuda.is_available(): exit("CUDA not available.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    model = torch.jit.load("dcgan_generator_scripted.pt")
    model.to(device)
    model.eval()

    noise = torch.randn(BATCH_SIZE, NOISE_DIM, 1, 1, device=device)

    print("Warming up...")
    with torch.no_grad():
        for _ in range(WARMUP_ITERATIONS): _ = model(noise)
    torch.cuda.synchronize()

    print("Starting benchmark...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for i in range(NUM_ITERATIONS): _ = model(noise)
    end_event.record()
    torch.cuda.synchronize()

    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / NUM_ITERATIONS

    print("\n--- PyTorch Results ---")
    print(f"Total time for {NUM_ITERATIONS} iterations: {total_time_ms:.3f} ms")
    print(f"Average inference time: {avg_time_ms:.4f} ms")