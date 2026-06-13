"""
train.py — Training Loop for the CNN
=====================================

This module implements the training pipeline — the process of teaching our
CNN to recognize images by showing it thousands of examples.

╔══════════════════════════════════════════════════════════════════════╗
║                 HOW TRAINING WORKS                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  For each EPOCH (full pass through the training data):               ║
║                                                                      ║
║    For each BATCH of images:                                         ║
║      1. FORWARD PASS  — Feed images through the CNN, get predictions ║
║      2. COMPUTE LOSS  — How wrong were the predictions?              ║
║      3. BACKWARD PASS — Calculate gradients (which direction to      ║
║                         adjust each weight to reduce the loss)       ║
║      4. UPDATE WEIGHTS — Optimizer adjusts weights using gradients   ║
║                                                                      ║
║  After each epoch:                                                   ║
║    - Validate on test set (NO gradient computation)                  ║
║    - Save the model if it's the best so far                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

KEY CONCEPTS:

LOSS FUNCTION (CrossEntropyLoss):
    Measures how far the model's predictions are from the true labels.
    - Perfect prediction → loss = 0
    - Random guessing (10 classes) → loss ≈ 2.3
    - Terrible prediction → loss is very high
    
    CrossEntropyLoss combines LogSoftmax + NLLLoss. It's the standard
    choice for multi-class classification.

OPTIMIZER (Adam):
    Controls HOW weights are updated using the gradients.
    Adam = "Adaptive Moment Estimation" — it automatically adjusts
    the learning rate for each parameter based on past gradients.
    Think of it as a smart hiker who adjusts step size based on the terrain.

GRADIENTS:
    Gradients tell us "if I increase this weight slightly, how much
    does the loss change?" The optimizer uses this to update weights
    in the direction that DECREASES the loss.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.config import (
    LEARNING_RATE, NUM_EPOCHS, DEVICE, MODEL_DIR,
    BEST_MODEL_PATH, set_seed, RANDOM_SEED
)
from src.dataset import get_train_loader, get_test_loader
from src.model import SimpleCNN, model_summary


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for ONE epoch (one pass through all training data).
    
    Args:
        model: The CNN to train
        loader: DataLoader with training batches
        criterion: Loss function (CrossEntropyLoss)
        optimizer: Weight update algorithm (Adam)
        device: Where to run computation (cuda/mps/cpu)
    
    Returns:
        avg_loss: Average loss across all batches
        accuracy: Training accuracy for this epoch (%)
    """
    model.train()  # Set model to training mode
                   # This ENABLES dropout and uses batch stats for BatchNorm
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        # Move data to the same device as the model
        images = images.to(device)
        labels = labels.to(device)
        
        # ---- STEP 1: Forward pass ----
        # Feed images through the CNN to get predictions
        outputs = model(images)  # Shape: (batch_size, 10)
        
        # ---- STEP 2: Compute loss ----
        # How wrong are we? Lower = better
        loss = criterion(outputs, labels)
        
        # ---- STEP 3: Backward pass ----
        optimizer.zero_grad()  # Clear old gradients (PyTorch accumulates them!)
        loss.backward()        # Compute gradients for ALL parameters
        
        # ---- STEP 4: Update weights ----
        optimizer.step()       # Adam adjusts weights using the gradients
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)  # Get class with highest score
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """
    Evaluate the model on validation/test data (NO gradient computation).
    
    IMPORTANT differences from training:
    - model.eval() — disables dropout, uses running stats for BatchNorm
    - torch.no_grad() — stops gradient computation, saves memory
    - No optimizer.step() — we're just measuring, not learning
    
    Args & Returns: Same as train_one_epoch
    """
    model.eval()  # Set to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No need to track gradients for evaluation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE):
    """
    Full training pipeline: create model, train, validate, save best.
    
    This is the main orchestrator that ties everything together.
    
    Args:
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate for Adam
        
    Returns:
        model: The trained CNN model
        history: Dict with training/validation loss and accuracy per epoch
    """
    # Set random seeds for reproducibility
    set_seed(RANDOM_SEED)
    
    # Create output directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Initialize components
    print("=" * 60)
    print("  TRAINING CNN IMAGE CLASSIFIER")
    print("=" * 60)
    
    # 1. Load data
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    
    # 2. Create model and move to device
    model = SimpleCNN().to(DEVICE)
    model_summary(model)
    
    # 3. Define loss function
    # CrossEntropyLoss: the standard for multi-class classification
    # It expects RAW logits (not softmax), and integer labels
    criterion = nn.CrossEntropyLoss()
    
    # 4. Define optimizer
    # Adam: adaptive learning rate optimizer — good default choice
    # It maintains per-parameter learning rates and momentum
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 5. Learning rate scheduler (optional but helpful)
    # Reduce LR by 10x when validation loss stops improving for 5 epochs
    # WHY: Start with big steps to learn quickly, then small steps to fine-tune
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    )
    
    # Track history for plotting later
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\n🏋️ Starting training for {num_epochs} epochs on {DEVICE}...\n")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, DEVICE
        )
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, BEST_MODEL_PATH)
            marker = " ⭐ BEST"
        else:
            marker = ""
        
        # Print progress
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch [{epoch:2d}/{num_epochs}] "
            f"│ Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% "
            f"│ Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% "
            f"│ LR: {current_lr:.6f} "
            f"│ {epoch_time:.1f}s{marker}"
        )
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    total_time = time.time() - start_time
    
    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"  ⏱️  Total time:    {total_time / 60:.1f} minutes")
    print(f"  🏆 Best val acc:   {best_val_acc:.2f}%")
    print(f"  💾 Model saved to: {BEST_MODEL_PATH}")
    print(f"{'=' * 60}")
    
    return model, history


# =============================================================================
# Run training directly
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"Number of epochs (default: {NUM_EPOCHS})")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"Learning rate (default: {LEARNING_RATE})")
    args = parser.parse_args()
    
    model, history = train_model(
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
