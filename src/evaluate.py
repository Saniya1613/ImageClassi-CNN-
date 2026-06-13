"""
evaluate.py — Model Evaluation & Metrics
=========================================

After training, we need to measure HOW WELL our model actually performs.
Just looking at accuracy isn't enough — we need to understand:

- Which classes does the model get right?
- Which classes does it confuse?
- Is the model biased toward certain classes?

KEY METRICS:

1. ACCURACY: % of images correctly classified
   - Simple but can be misleading with imbalanced datasets
   - CIFAR-10 is balanced (1000 test images per class), so accuracy is OK here

2. CONFUSION MATRIX: An N×N grid showing:
   - Row = true class, Column = predicted class
   - Diagonal = correct predictions
   - Off-diagonal = mistakes
   - Example: If cell [cat, dog] is high, the model often confuses cats for dogs

3. CLASSIFICATION REPORT: Per-class metrics:
   - Precision: Of all images predicted as "cat", how many actually are cats?
   - Recall: Of all actual cats, how many did the model find?
   - F1-Score: Harmonic mean of precision and recall (balances both)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from src.config import (
    DEVICE, BEST_MODEL_PATH, CIFAR10_CLASSES, NUM_CLASSES
)
from src.dataset import get_test_loader
from src.model import SimpleCNN


def evaluate_model(model, test_loader, device=DEVICE):
    """
    Run the model on the entire test set and collect predictions.
    
    Args:
        model: Trained CNN model
        test_loader: DataLoader with test data
        device: Computation device
        
    Returns:
        all_preds: numpy array of predicted labels
        all_labels: numpy array of true labels
        all_probs: numpy array of prediction probabilities (for each class)
    """
    model.eval()  # Evaluation mode (no dropout, fixed BatchNorm)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            
            # Get predicted class (highest probability)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def print_classification_report(y_true, y_pred):
    """
    Print per-class precision, recall, and F1-score.
    
    Example output:
                  precision    recall  f1-score   support
        airplane       0.82      0.85      0.83      1000
        automobile     0.90      0.93      0.91      1000
        ...
    
    What these mean:
    - Precision: "When the model says 'airplane', is it right?"
    - Recall: "Of all airplanes, how many did the model find?"
    - F1: Balance of precision and recall
    - Support: Number of test images for this class
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CIFAR10_CLASSES,
        digits=4,
    )
    
    overall_acc = accuracy_score(y_true, y_pred)
    
    print("=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print(f"  Overall Accuracy: {overall_acc * 100:.2f}%")
    print("=" * 60)
    
    return report


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """
    Create a beautiful heatmap showing what the model confuses.
    
    HOW TO READ IT:
    - Each row represents the ACTUAL class
    - Each column represents the PREDICTED class
    - The diagonal (top-left to bottom-right) = CORRECT predictions
    - Off-diagonal values = MISTAKES
    
    Dark colors = low count, bright colors = high count
    Ideally, only the diagonal should be bright!
    
    Common confusion patterns in CIFAR-10:
    - cat ↔ dog (both are furry animals)
    - automobile ↔ truck (both are vehicles)
    - airplane ↔ ship (both have similar backgrounds — sky/water)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,           # Show numbers in each cell
        fmt="d",              # Integer format
        cmap="Blues",          # Blue color scheme
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        linewidths=0.5,
        linecolor="gray",
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Confusion matrix saved to {save_path}")


def plot_per_class_accuracy(y_true, y_pred, save_path="per_class_accuracy.png"):
    """
    Bar chart showing accuracy for each individual class.
    
    This helps identify which classes are hardest for the model.
    Usually: animals are harder than vehicles in CIFAR-10.
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    
    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(per_class_acc)
    sorted_classes = [CIFAR10_CLASSES[i] for i in sorted_indices]
    sorted_acc = per_class_acc[sorted_indices]
    
    # Color bars: red for low, green for high
    colors = plt.cm.RdYlGn(sorted_acc / 100)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(sorted_classes, sorted_acc, color=colors, edgecolor="gray")
    
    # Add value labels on bars
    for bar, acc in zip(bars, sorted_acc):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc:.1f}%", va="center", fontweight="bold")
    
    plt.xlabel("Accuracy (%)", fontsize=12)
    plt.title("Per-Class Accuracy", fontsize=16, fontweight="bold")
    plt.xlim(0, 105)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Per-class accuracy saved to {save_path}")


def load_and_evaluate():
    """
    Load the best saved model and run full evaluation.
    
    This is what you run after training is complete to see final results.
    """
    print("=" * 60)
    print("  EVALUATING BEST MODEL")
    print("=" * 60)
    
    # Load test data
    test_loader = get_test_loader()
    
    # Create model and load saved weights
    model = SimpleCNN().to(DEVICE)
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"\n📂 Loaded model from: {BEST_MODEL_PATH}")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    print(f"   Best val accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Run evaluation
    y_pred, y_true, y_probs = evaluate_model(model, test_loader)
    
    # Print detailed report
    print_classification_report(y_true, y_pred)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred)
    
    # Plot per-class accuracy
    plot_per_class_accuracy(y_true, y_pred)
    
    return model, y_pred, y_true, y_probs


# =============================================================================
# Run evaluation directly
# =============================================================================
if __name__ == "__main__":
    load_and_evaluate()
