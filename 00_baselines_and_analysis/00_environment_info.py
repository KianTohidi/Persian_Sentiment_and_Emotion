"""Print Python, library and GPU information for the current environment."""
import importlib.metadata as metadata
import sys

print("Python:", sys.version.split()[0])
for package in ["transformers", "torch", "accelerate", "scikit-learn", "scipy", "pandas", "numpy",
                "openpyxl", "matplotlib"]:
    try:
        print(f"{package}: {metadata.version(package)}")
    except metadata.PackageNotFoundError:
        print(f"{package}: not installed")

import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("CUDA version used by PyTorch:", torch.version.cuda)
