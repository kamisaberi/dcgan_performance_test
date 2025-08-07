 
# export_weights.py
import torch
import torch.nn as nn
import numpy as np
import os

# --- Configuration ---
output_dir_bin = "weights"
output_file_pt = "dcgan_generator_traced.pt"
noise_dim = 100
batch_size = 1 # We trace with batch size 1 for single-image inference

# --- Create Directories ---
os.makedirs(output_dir_bin, exist_ok=True)
print(f"Binary weights will be saved in the '{output_dir_bin}/' directory.")

# --- Load Model ---
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=False)
netG = model.netG
netG.eval()

print("Generator Model Architecture:")
print(netG)

# ===================================================================
# Part 1: Export TorchScript model for LibTorch
# ===================================================================
print(f"\n--- Exporting TorchScript model to {output_file_pt} ---")
# Create a dummy input tensor with the correct shape for tracing
example_input = torch.randn(batch_size, noise_dim, 1, 1)
try:
    traced_model = torch.jit.trace(netG, example_input)
    traced_model.save(output_file_pt)
    print("TorchScript model saved successfully.")
except Exception as e:
    print(f"Error tracing model: {e}")

# ===================================================================
# Part 2: Export raw binary weights for custom CUDA project
# ===================================================================
print(f"\n--- Exporting raw binary weights to {output_dir_bin}/ ---")
# The names must match what the C++ code expects
layer_map = {
    'main.0': 'layer1', 'main.1': 'layer1_bn',
    'main.3': 'layer2', 'main.4': 'layer2_bn',
    'main.6': 'layer3', 'main.7': 'layer3_bn',
    'main.9': 'layer4', 'main.10': 'layer4_bn',
    'main.12': 'layer5',
}

def save_param(tensor, filename):
    tensor.detach().cpu().numpy().astype(np.float32).tofile(filename)
    print(f"Saved {filename} (shape: {tensor.shape})")

for name, module in netG.named_modules():
    if name in layer_map:
        cpp_layer_name = layer_map[name]
        
        if isinstance(module, nn.ConvTranspose2d):
            save_param(module.weight, os.path.join(output_dir_bin, f"{cpp_layer_name}_weight.bin"))
        
        elif isinstance(module, nn.BatchNorm2d):
            save_param(module.weight, os.path.join(output_dir_bin, f"{cpp_layer_name}_gamma.bin"))
            save_param(module.bias, os.path.join(output_dir_bin, f"{cpp_layer_name}_beta.bin"))
            save_param(module.running_mean, os.path.join(output_dir_bin, f"{cpp_layer_name}_mean.bin"))
            save_param(module.running_var, os.path.join(output_dir_bin, f"{cpp_layer_name}_var.bin"))

print("\nAll exports complete.") 