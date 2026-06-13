"""
transfer_learn.py — Transfer Learning with Pretrained ResNet-18
================================================================

╔══════════════════════════════════════════════════════════════════════╗
║                 WHAT IS TRANSFER LEARNING?                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Instead of training a CNN from scratch (random weights), we take    ║
║  a model that was ALREADY trained on a huge dataset (ImageNet,       ║
║  14 million images, 1000 classes) and adapt it for our task.         ║
║                                                                      ║
║  WHY DOES THIS WORK?                                                 ║
║  Early CNN layers learn UNIVERSAL features:                          ║
║    - Layer 1: Edges, gradients                                       ║
║    - Layer 2: Textures, patterns                                     ║
║    - Layer 3: Object parts                                           ║
║  These features are useful for ANY image task, not just ImageNet!    ║
║                                                                      ║
║  By reusing these pretrained features, we:                           ║
║    ✅ Need MUCH less training data                                   ║
║    ✅ Train MUCH faster                                              ║
║    ✅ Get HIGHER accuracy                                            ║
║                                                                      ║
║  ANALOGY: It's like a professional chef (trained on French cuisine)  ║
║  learning Japanese cooking. They don't start from zero — they        ║
║  already know knife skills, heat control, flavor balancing.          ║
║  They just need to learn the new recipes.                            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                 TWO APPROACHES                                       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. FEATURE EXTRACTION (freeze everything, train only the final FC)  ║
║     - Fast, works well when your data is similar to ImageNet         ║
║     - The pretrained layers act as a fixed feature extractor         ║
║                                                                      ║
║  2. FINE-TUNING (unfreeze everything, train with lower learning rate)║
║     - Slower but usually better accuracy                             ║
║     - Adjusts all layers slightly for your specific task             ║
║     - Use a LOWER learning rate to avoid destroying pretrained       ║
║       features (0.0001 instead of 0.001)                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

RESNET-18 ARCHITECTURE:
    ResNet = "Residual Network" — uses SKIP CONNECTIONS that let gradients
    flow directly through the network, enabling much deeper architectures.
    
    ResNet-18 has 18 layers (hence the name) with ~11 million parameters.
    Compare to our SimpleCNN with ~600K parameters — ResNet is ~18x bigger!
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from src.config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, MODEL_DIR,
    TRANSFER_MODEL_PATH, set_seed, RANDOM_SEED
)
from src.dataset import get_train_loader, get_test_loader
from src.train import train_one_epoch, validate


def create_transfer_model(freeze_backbone=True):
    """
    Create a ResNet-18 model adapted for CIFAR-10 classification.
    
    Steps:
    1. Load ResNet-18 with pretrained ImageNet weights
    2. Replace the final fully connected layer (1000 classes → 10 classes)
    3. Optionally freeze all layers except the final FC
    
    Args:
        freeze_backbone: If True, only train the final FC layer (feature extraction).
                        If False, train all layers (fine-tuning).
    
    Returns:
        model: Modified ResNet-18 ready for CIFAR-10
    """
    # Load pretrained ResNet-18
    # 'IMAGENET1K_V1' = weights trained on ImageNet (1.2M images, 1000 classes)
    print("📥 Loading pretrained ResNet-18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # =========================================================================
    # FREEZE / UNFREEZE LAYERS
    # =========================================================================
    if freeze_backbone:
        # Freeze ALL parameters — they won't be updated during training
        # This is FEATURE EXTRACTION mode
        for param in model.parameters():
            param.requires_grad = False
        print("🧊 Backbone frozen — only training final FC layer")
    else:
        print("🔥 All layers unfrozen — fine-tuning entire network")
    
    # =========================================================================
    # REPLACE FINAL LAYER
    # =========================================================================
    # ResNet-18's original final layer: Linear(512, 1000) for ImageNet
    # We replace it with: Linear(512, 10) for CIFAR-10
    
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, NUM_CLASSES),
    )
    # The new fc layer's parameters require_grad=True by default,
    # so it will always be trained, even when backbone is frozen.
    
    print(f"   Replaced FC: {num_features} → {NUM_CLASSES} classes")
    
    # Count trainable parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params:     {total:,}")
    print(f"   Trainable params: {trainable:,} ({100*trainable/total:.1f}%)")
    
    return model


def train_transfer_model(mode="fine_tune", num_epochs=NUM_EPOCHS):
    """
    Train a transfer learning model on CIFAR-10.
    
    Args:
        mode: "feature_extract" or "fine_tune"
        num_epochs: Number of training epochs
        
    Returns:
        model: Trained model
        history: Training history dict
    """
    set_seed(RANDOM_SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 60)
    print(f"  TRANSFER LEARNING — {mode.upper()}")
    print("=" * 60)
    
    # Load data
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    
    # Create model
    freeze = (mode == "feature_extract")
    model = create_transfer_model(freeze_backbone=freeze).to(DEVICE)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer — different learning rates for different modes
    if mode == "feature_extract":
        # Only optimize the final FC layer (the only unfrozen layer)
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        # Fine-tuning: use a LOWER learning rate to preserve pretrained features
        # We use different LRs for different parts:
        # - Backbone: very low LR (0.0001) — gentle updates
        # - New FC: higher LR (0.001) — needs to learn from scratch
        optimizer = optim.Adam([
            {"params": [p for n, p in model.named_parameters()
                       if "fc" not in n and p.requires_grad],
             "lr": 0.0001},  # Backbone: low LR
            {"params": model.fc.parameters(),
             "lr": 0.001},   # New FC: higher LR
        ])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )
    
    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    print(f"\n🏋️ Training for {num_epochs} epochs on {DEVICE}...\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate(
            model, test_loader, criterion, DEVICE
        )
        
        scheduler.step(val_loss)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "mode": mode,
            }, TRANSFER_MODEL_PATH)
            marker = " ⭐ BEST"
        else:
            marker = ""
        
        epoch_time = time.time() - epoch_start
        print(
            f"Epoch [{epoch:2d}/{num_epochs}] "
            f"│ Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% "
            f"│ Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% "
            f"│ {epoch_time:.1f}s{marker}"
        )
    
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"  TRANSFER LEARNING COMPLETE ({mode})")
    print(f"{'=' * 60}")
    print(f"  ⏱️  Total time:    {total_time / 60:.1f} minutes")
    print(f"  🏆 Best val acc:   {best_val_acc:.2f}%")
    print(f"  💾 Model saved to: {TRANSFER_MODEL_PATH}")
    print(f"{'=' * 60}")
    
    return model, history


# =============================================================================
# Run transfer learning directly
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfer Learning on CIFAR-10")
    parser.add_argument("--mode", type=str, default="fine_tune",
                        choices=["feature_extract", "fine_tune"],
                        help="Transfer learning mode (default: fine_tune)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs (default: 10)")
    args = parser.parse_args()
    
    model, history = train_transfer_model(
        mode=args.mode,
        num_epochs=args.epochs,
    )
