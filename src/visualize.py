"""
visualize.py — Training Curves & Prediction Visualization
==========================================================

Visualization is crucial for understanding model behavior:

1. TRAINING CURVES — Show how loss and accuracy change over epochs
   - Is the model learning? (loss should decrease)
   - Is it overfitting? (training acc goes up but validation acc stalls)
   - Is the learning rate right? (loss should decrease smoothly, not jump)

2. PREDICTION VISUALIZATION — See actual model outputs
   - Green border = correct prediction ✅
   - Red border = wrong prediction ❌
   - Helps build intuition about what the model struggles with

3. MISCLASSIFICATION ANALYSIS — Focus on mistakes
   - What does the model get wrong?
   - Are the mistakes reasonable? (cat mistaken for dog = OK,
     cat mistaken for airplane = something is broken)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.config import DEVICE, CIFAR10_CLASSES, BEST_MODEL_PATH
from src.dataset import get_test_loader, denormalize
from src.model import SimpleCNN


def plot_training_curves(history, save_path="training_curves.png"):
    """
    Plot training and validation loss/accuracy over epochs.
    
    WHAT TO LOOK FOR:
    
    ✅ Good training:
       - Both train and val loss decrease
       - Both accuracies increase
       - Small gap between train and val curves
    
    ⚠️ Overfitting:
       - Train loss keeps decreasing, but val loss starts INCREASING
       - Train accuracy is much higher than val accuracy
       - The gap between curves widens over epochs
       → Fix: Add more dropout, data augmentation, or stop earlier
    
    ⚠️ Underfitting:
       - Both train and val loss are high
       - Both accuracies are low
       → Fix: Bigger model, more epochs, higher learning rate
    
    Args:
        history: Dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Where to save the plot
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training Progress", fontsize=16, fontweight="bold")
    
    # --- Loss Plot ---
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4,
             label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=4,
             label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(epochs) + 1)
    
    # --- Accuracy Plot ---
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=4,
             label="Train Acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"], "r-o", markersize=4,
             label="Val Acc", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.set_title("Accuracy Curves", fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, len(epochs) + 1)
    
    # Highlight best validation accuracy
    best_epoch = np.argmax(history["val_acc"]) + 1
    best_acc = max(history["val_acc"])
    ax2.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.5)
    ax2.annotate(
        f"Best: {best_acc:.1f}%\n(Epoch {best_epoch})",
        xy=(best_epoch, best_acc),
        xytext=(best_epoch + 1, best_acc - 5),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=10, color="green", fontweight="bold",
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Training curves saved to {save_path}")


def show_predictions(model, test_loader, device=DEVICE, num_images=16,
                     save_path="predictions.png"):
    """
    Show a grid of test images with predicted vs true labels.
    
    - GREEN title = correct prediction ✅
    - RED title = wrong prediction ❌
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        device: Computation device
        num_images: Number of images to display
        save_path: Where to save the plot
    """
    model.eval()
    
    # Get one batch of images
    images, labels = next(iter(test_loader))
    images_device = images.to(device)
    
    with torch.no_grad():
        outputs = model(images_device)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(1)
    
    # Set up grid
    cols = 4
    rows = num_images // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle("Model Predictions on Test Images",
                 fontsize=16, fontweight="bold")
    
    for idx in range(num_images):
        row, col = idx // cols, idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Denormalize and convert for display
        img = denormalize(images[idx])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        pred_label = CIFAR10_CLASSES[preds[idx]]
        true_label = CIFAR10_CLASSES[labels[idx]]
        conf = confidences[idx].item() * 100
        
        # Green for correct, red for wrong
        is_correct = preds[idx] == labels[idx]
        color = "green" if is_correct else "red"
        symbol = "✅" if is_correct else "❌"
        
        ax.set_title(
            f"{symbol} {pred_label} ({conf:.0f}%)\n[True: {true_label}]",
            fontsize=9, color=color, fontweight="bold",
        )
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Predictions saved to {save_path}")


def show_misclassified(model, test_loader, device=DEVICE, num_images=16,
                       save_path="misclassified.png"):
    """
    Show images that the model got WRONG.
    
    This is arguably the most useful visualization because:
    - It reveals systematic weaknesses (e.g., always confusing cats/dogs)
    - It can reveal data quality issues (some CIFAR-10 images are ambiguous)
    - It helps you decide what to improve next
    
    Args:
        model: Trained model
        test_loader: DataLoader with test data
        device: Computation device
        num_images: Number of misclassified images to show
        save_path: Where to save the plot
    """
    model.eval()
    
    wrong_images = []
    wrong_preds = []
    wrong_labels = []
    wrong_confs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            outputs = model(images_device)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(1)
            
            # Find misclassified images
            mask = preds.cpu() != labels
            if mask.any():
                wrong_images.extend(images[mask])
                wrong_preds.extend(preds.cpu()[mask])
                wrong_labels.extend(labels[mask])
                wrong_confs.extend(confidences.cpu()[mask])
            
            if len(wrong_images) >= num_images:
                break
    
    # Limit to requested number
    num_show = min(num_images, len(wrong_images))
    
    if num_show == 0:
        print("🎉 No misclassified images found! Perfect model!")
        return
    
    cols = 4
    rows = max(1, num_show // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    fig.suptitle("❌ Misclassified Images — What the Model Got Wrong",
                 fontsize=14, fontweight="bold", color="red")
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for idx in range(num_show):
        ax = axes_flat[idx]
        
        img = denormalize(wrong_images[idx])
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        pred_label = CIFAR10_CLASSES[wrong_preds[idx]]
        true_label = CIFAR10_CLASSES[wrong_labels[idx]]
        conf = wrong_confs[idx].item() * 100
        
        ax.set_title(
            f"Pred: {pred_label} ({conf:.0f}%)\nTrue: {true_label}",
            fontsize=9, color="red", fontweight="bold",
        )
        ax.axis("off")
    
    # Hide empty subplots
    for idx in range(num_show, len(axes_flat)):
        axes_flat[idx].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Misclassified images saved to {save_path}")


# =============================================================================
# Run visualization directly
# =============================================================================
if __name__ == "__main__":
    print("Loading best model for visualization...")
    
    model = SimpleCNN().to(DEVICE)
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loader = get_test_loader()
    
    show_predictions(model, test_loader)
    show_misclassified(model, test_loader)
