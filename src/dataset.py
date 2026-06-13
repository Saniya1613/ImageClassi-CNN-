"""
dataset.py — Data Loading & Augmentation Pipeline
==================================================

This module handles everything related to getting CIFAR-10 data ready for
our CNN. There are three key stages:

1. DOWNLOADING: PyTorch auto-downloads CIFAR-10 on first run (~170MB)

2. TRANSFORMS (Preprocessing):
   - Convert PIL images to PyTorch tensors (pixel values: 0-255 → 0.0-1.0)
   - Normalize using CIFAR-10 mean/std (centers data around 0)

3. DATA AUGMENTATION (Training only!):
   - Randomly flip images horizontally (a cat flipped is still a cat)
   - Randomly crop with padding (shifts the object around in the frame)
   
   WHY AUGMENT?
   Augmentation artificially increases your training data diversity.
   Instead of seeing the same 50k images over and over, the model sees
   slightly different versions each epoch. This prevents OVERFITTING
   (memorizing training data instead of learning general patterns).
   
   IMPORTANT: We NEVER augment test data! Test data must be consistent
   so we can fairly measure model performance.

4. DATALOADERS:
   - Wrap datasets in DataLoaders for efficient batched loading
   - Shuffle training data (so the model doesn't learn the order)
   - Don't shuffle test data (order doesn't matter for evaluation)
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, NUM_WORKERS, DATA_DIR,
    CIFAR10_MEAN, CIFAR10_STD, CIFAR10_CLASSES,
    IMAGE_SIZE
)


def get_train_transforms():
    """
    Build the training transform pipeline.
    
    The transforms are applied IN ORDER:
    1. RandomHorizontalFlip — 50% chance to mirror the image left-right
       WHY: Objects look the same when flipped (a dog facing left = dog facing right)
       
    2. RandomCrop(32, padding=4) — Pad 4 pixels on each side, then randomly
       crop back to 32x32. This shifts the object slightly in the frame.
       WHY: Teaches the model that objects can appear anywhere in the image
       
    3. ToTensor — Converts PIL Image (H×W×C, 0-255) to PyTorch tensor (C×H×W, 0.0-1.0)
       WHY: Neural networks work with tensors, not images
       
    4. Normalize — Subtract mean, divide by std for each RGB channel
       WHY: Centers data around 0 with unit variance → faster, more stable training
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(IMAGE_SIZE, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_test_transforms():
    """
    Build the test/validation transform pipeline.
    
    NO augmentation here — just convert to tensor and normalize.
    We want a consistent, fair evaluation every time.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def get_train_loader():
    """
    Download CIFAR-10 training data and wrap it in a DataLoader.
    
    Returns:
        DataLoader: Yields batches of (images, labels) tensors
        
    What's a DataLoader?
        Instead of loading ALL 50,000 images into memory at once,
        a DataLoader gives you small batches (e.g., 64 images at a time).
        This is memory-efficient and allows the model to update its
        weights more frequently (64 updates per epoch vs 1).
    """
    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,           # Use the 50k training images
        download=True,        # Download if not already present
        transform=get_train_transforms(),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,         # Shuffle each epoch so the model doesn't
                              # memorize the order of images
        num_workers=NUM_WORKERS,
        pin_memory=True,      # Speed up CPU→GPU transfer
    )
    
    print(f"📦 Training data: {len(train_dataset)} images, "
          f"{len(train_loader)} batches of {BATCH_SIZE}")
    
    return train_loader


def get_test_loader():
    """
    Download CIFAR-10 test data and wrap it in a DataLoader.
    
    Returns:
        DataLoader: Yields batches of (images, labels) tensors
    """
    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,          # Use the 10k test images
        download=True,
        transform=get_test_transforms(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,        # No need to shuffle test data
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    
    print(f"📦 Test data: {len(test_dataset)} images, "
          f"{len(test_loader)} batches of {BATCH_SIZE}")
    
    return test_loader


def denormalize(tensor):
    """
    Reverse the normalization so we can display images properly.
    
    The model sees normalized images (centered around 0), but humans
    need pixel values in 0-1 range to see the actual image.
    
    Math: original = (normalized × std) + mean
    """
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    return tensor * std + mean


def show_sample_images(loader, num_images=16):
    """
    Display a grid of sample images from a DataLoader.
    
    This is useful to:
    - Verify data loaded correctly
    - See what augmented images look like
    - Understand what the model is working with
    
    Args:
        loader: DataLoader to sample from
        num_images: Number of images to display (will be arranged in a grid)
    """
    # Get one batch of images
    images, labels = next(iter(loader))
    
    # Set up the grid
    cols = 4
    rows = num_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.5 * rows))
    fig.suptitle("Sample CIFAR-10 Images", fontsize=16, fontweight="bold")
    
    for idx in range(num_images):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Denormalize for display
        img = denormalize(images[idx])
        
        # Convert from (C, H, W) → (H, W, C) for matplotlib
        img = img.permute(1, 2, 0).numpy()
        
        # Clip to valid range (numerical precision may cause tiny overshoots)
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        ax.set_title(CIFAR10_CLASSES[labels[idx]], fontsize=11)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("💾 Saved sample grid to sample_images.png")


# =============================================================================
# Run this file directly to test the data pipeline
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  Testing Data Pipeline")
    print("=" * 50)
    
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    
    # Inspect one batch
    images, labels = next(iter(train_loader))
    print(f"\n📐 Batch shape: {images.shape}")
    print(f"   → {images.shape[0]} images")
    print(f"   → {images.shape[1]} channels (RGB)")
    print(f"   → {images.shape[2]}×{images.shape[3]} pixels")
    print(f"📋 Labels shape: {labels.shape}")
    print(f"   → First 10 labels: {[CIFAR10_CLASSES[l] for l in labels[:10].tolist()]}")
    
    # Show some images
    show_sample_images(train_loader)
