import torch
import torch.nn as nn
import numpy as np
import os

# --- Configuration ---
output_dir_bin = "weights"
output_file_pt = "dcgan_generator_scripted.pt"
noise_dim = 100
batch_size = 1

# --- Create Directories ---
os.makedirs(output_dir_bin, exist_ok=True)
print(f"Binary weights will be saved in the '{output_dir_bin}/' directory.")

# ===================================================================
# THE DEFINITIVE FIX: Define the Generator architecture directly
# This removes all dependencies on torch.hub and guarantees correctness.
# ===================================================================
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Instantiate our local, well-defined model
netG = Generator(nz=noise_dim, ngf=128, nc=3) # Use ngf=128 for a larger, more realistic model
netG.eval()

print("Generator Model Architecture:")
print(netG)

# ===================================================================
# Part 1: Export TorchScript model
# ===================================================================
print(f"\n--- Exporting TorchScript model to {output_file_pt} ---")
example_input = torch.randn(batch_size, noise_dim, 1, 1)
try:
    traced_model = torch.jit.trace(netG, example_input)
    traced_model.save(output_file_pt)
    print("TorchScript model saved successfully.")
except Exception as e:
    print(f"Error tracing model: {e}")

# ===================================================================
# Part 2: Export raw binary weights
# ===================================================================
print(f"\n--- Exporting raw binary weights to {output_dir_bin}/ ---")
# This mapping is now guaranteed to be correct.
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