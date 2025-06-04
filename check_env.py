# check_env.py
# I wrote this to make sure my laptop actually uses the RTX 4050.
# Training a U-Net on a CPU would take days, so this check is mandatory.

import sys
import importlib.util

def check_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    spec = importlib.util.find_spec(import_name)
    if spec is None:
        print(f"[-] Missing: {package_name} is not installed.")
        return False
    else:
        print(f"[+] Installed: {package_name}")
        return True

print("=== CHECKING MY ENVIRONMENT ===")
required = {
    "torch": "torch",
    "torchvision": "torchvision",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "tqdm": "tqdm",
    "tifffile": "tifffile",
    "pandas": "pandas",
    "scikit-learn": "sklearn"
}

missing = [pkg for pkg, imp in required.items() if not check_package(pkg, imp)]

try:
    import torch
    if torch.cuda.is_available():
        print(f"\n[INFO] GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("\n[!] WARNING: Running on CPU. I need to fix my CUDA installation.")
except Exception as e:
    print(f"[-] GPU check failed: {e}")

if missing:
    print(f"\n[!] Run this command first: pip install -r requirements.txt")
else:
    print("\n[+] My environment is perfect. Ready to build.")