"""
config.py — Centralized Configuration for the CNN Image Classifier
=================================================================

WHY THIS FILE EXISTS:
    Instead of scattering magic numbers (learning rate = 0.001, batch size = 64)
    across multiple files, we put ALL configuration in one place.
    
    This makes it easy to:
    - Experiment with different hyperparameters
    - Reproduce results (just share this file)
    - Understand what settings were used for a given training run

KEY CONCEPTS:
    - Hyperparameters: Settings that control how the model trains (learning rate,
      batch size, epochs). These are NOT learned by the model — YOU choose them.
    - Normalization values: The mean and std of CIFAR-10 pixel values. We use
      these to scale inputs to have zero mean and unit variance, which helps
      the model train faster and more stably.
    - Device: We auto-detect the best available hardware (GPU > MPS > CPU).
"""

import torch
import os

# =============================================================================
# HYPERPARAMETERS — Tune these to experiment!
# =============================================================================

# Batch size: How many images the model sees before updating its weights.
# Larger = faster training but more memory. 64 is a good default.
BATCH_SIZE = 64

# Learning rate: How big of a step the optimizer takes when updating weights.
# Too high → training is unstable (loss jumps around)
# Too low → training is very slow
# 0.001 is a safe starting point for Adam optimizer.
LEARNING_RATE = 0.001

# Number of epochs: How many times the model sees the ENTIRE training dataset.
# More epochs = more learning, but too many → overfitting.
NUM_EPOCHS = 20

# Number of workers for data loading (parallel data preprocessing).
# Set to 0 if you get multiprocessing errors.
NUM_WORKERS = 2

# =============================================================================
# DATASET CONSTANTS
# =============================================================================

# CIFAR-10 has exactly 10 classes
NUM_CLASSES = 10

# Each CIFAR-10 image is 32x32 pixels with 3 color channels (RGB)
IMAGE_SIZE = 32
NUM_CHANNELS = 3

# Human-readable class names — index matches the label number
# e.g., label 0 = "airplane", label 3 = "cat"
CIFAR10_CLASSES = [
    "airplane",     # 0
    "automobile",   # 1
    "bird",         # 2
    "cat",          # 3
    "deer",         # 4
    "dog",          # 5
    "frog",         # 6
    "horse",        # 7
    "ship",         # 8
    "truck",        # 9
]

# Normalization values for CIFAR-10
# These are the per-channel mean and standard deviation of all pixel values
# in the CIFAR-10 training set. We precompute them so every image gets
# the same normalization.
#
# WHY NORMALIZE?
#   Raw pixel values are 0-255. Neural networks work much better when inputs
#   are small numbers centered around 0. Normalizing to zero mean and unit
#   variance ensures no single channel dominates and gradients flow smoothly.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)  # R, G, B channel means
CIFAR10_STD = (0.2470, 0.2435, 0.2616)   # R, G, B channel stds

# =============================================================================
# PATHS
# =============================================================================

# Root directory of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where CIFAR-10 data will be downloaded
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Where trained model weights are saved
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Path for the best model checkpoint
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

# Path for transfer learning model
TRANSFER_MODEL_PATH = os.path.join(MODEL_DIR, "transfer_resnet18.pt")

# =============================================================================
# DEVICE DETECTION — Automatically use the best available hardware
# =============================================================================
# Priority: CUDA (NVIDIA GPU) → MPS (Apple Silicon) → CPU
#
# WHY GPUS?
#   CNNs do MILLIONS of matrix multiplications. GPUs have thousands of cores
#   designed for exactly this. Training on GPU can be 10-50x faster than CPU.

def get_device():
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple Silicon (MPS) acceleration")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU (training will be slower)")
    return device

DEVICE = get_device()

# =============================================================================
# REPRODUCIBILITY — Set seeds for consistent results
# =============================================================================
# Neural network training involves randomness (weight initialization, data
# shuffling, dropout). Setting seeds makes results reproducible.

RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# Print config summary when this module is imported
if __name__ == "__main__":
    print("=" * 50)
    print("  CNN Image Classifier — Configuration")
    print("=" * 50)
    print(f"  Batch Size:     {BATCH_SIZE}")
    print(f"  Learning Rate:  {LEARNING_RATE}")
    print(f"  Epochs:         {NUM_EPOCHS}")
    print(f"  Num Classes:    {NUM_CLASSES}")
    print(f"  Image Size:     {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Device:         {DEVICE}")
    print(f"  Data Dir:       {DATA_DIR}")
    print(f"  Model Dir:      {MODEL_DIR}")
    print("=" * 50)
