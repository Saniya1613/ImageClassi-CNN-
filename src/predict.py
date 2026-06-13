"""
predict.py — Inference on Custom Images
========================================

This module lets you classify ANY image using your trained model.

WORKFLOW:
    1. Load your trained model (best_model.pt)
    2. Load and preprocess any image from disk
    3. Run it through the model
    4. Get the predicted class and confidence

PREPROCESSING IS CRITICAL:
    The model was trained on CIFAR-10 images that were:
    - 32×32 pixels
    - Normalized with CIFAR-10 mean/std
    
    So when we predict on a new image, we MUST apply the SAME preprocessing,
    otherwise the model will get garbage input and give garbage output.
    
    Steps:
    1. Resize to 32×32 (CIFAR-10 resolution)
    2. Convert to tensor (0-255 → 0.0-1.0)
    3. Normalize with CIFAR-10 mean/std
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.config import (
    DEVICE, BEST_MODEL_PATH, CIFAR10_CLASSES,
    CIFAR10_MEAN, CIFAR10_STD, IMAGE_SIZE
)
from src.model import SimpleCNN


def get_prediction_transform():
    """
    Transform pipeline for custom images.
    
    Must match the test transforms used during training:
    1. Resize to 32×32 (CIFAR-10 resolution)
    2. Convert to tensor
    3. Normalize with CIFAR-10 stats
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def load_trained_model(model_path=BEST_MODEL_PATH):
    """
    Load the best trained model from disk.
    
    Returns:
        model: Loaded and ready-to-use CNN model in eval mode
    """
    model = SimpleCNN().to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set to evaluation mode (no dropout, fixed BatchNorm)
    
    print(f"📂 Loaded model from: {model_path}")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    print(f"   Val accuracy: {checkpoint['val_acc']:.2f}%")
    
    return model


def predict_image(model, image_path, show=True):
    """
    Classify a single image and show the result.
    
    Args:
        model: Trained CNN model
        image_path: Path to the image file (any format: jpg, png, etc.)
        show: Whether to display the image with prediction
        
    Returns:
        predicted_class: String name of the predicted class
        confidence: Confidence score (0-100%)
        all_probs: Dict of class → probability for all classes
    """
    # Load and preprocess the image
    transform = get_prediction_transform()
    
    # Open image and convert to RGB (handles grayscale, RGBA, etc.)
    image = Image.open(image_path).convert("RGB")
    
    # Apply transforms: resize → tensor → normalize
    input_tensor = transform(image)
    
    # Add batch dimension: (3, 32, 32) → (1, 3, 32, 32)
    # The model expects a BATCH of images, even for a single image
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Run prediction
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = F.softmax(outputs, dim=1)[0]  # Convert logits → probs
    
    # Get top prediction
    confidence, predicted_idx = probabilities.max(0)
    predicted_class = CIFAR10_CLASSES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    
    # Get all class probabilities
    all_probs = {
        CIFAR10_CLASSES[i]: probabilities[i].item() * 100
        for i in range(len(CIFAR10_CLASSES))
    }
    
    # Sort by probability (highest first)
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    
    if show:
        _display_prediction(image, predicted_class, confidence_pct, all_probs)
    
    return predicted_class, confidence_pct, all_probs


def _display_prediction(image, predicted_class, confidence, all_probs):
    """
    Display the image with prediction results and a probability bar chart.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1, 1.5]})
    
    # Left: Show the image
    ax1.imshow(image)
    ax1.set_title(
        f"Prediction: {predicted_class.upper()}\n"
        f"Confidence: {confidence:.1f}%",
        fontsize=14, fontweight="bold",
        color="green" if confidence > 50 else "orange",
    )
    ax1.axis("off")
    
    # Right: Probability bar chart for all classes
    classes = list(all_probs.keys())
    probs = list(all_probs.values())
    
    # Color the top prediction differently
    colors = ["#2ecc71" if c == predicted_class else "#3498db" for c in classes]
    
    bars = ax2.barh(classes[::-1], probs[::-1], color=colors[::-1],
                    edgecolor="white", linewidth=0.5)
    
    # Add value labels
    for bar, prob in zip(bars, probs[::-1]):
        if prob > 2:
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{prob:.1f}%", va="center", fontsize=9)
    
    ax2.set_xlabel("Probability (%)", fontsize=11)
    ax2.set_title("Class Probabilities", fontsize=14, fontweight="bold")
    ax2.set_xlim(0, max(probs) * 1.2)
    ax2.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("prediction_result.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"💾 Prediction result saved to prediction_result.png")


def predict_batch(model, image_paths):
    """
    Classify multiple images at once.
    
    Args:
        model: Trained CNN model
        image_paths: List of paths to image files
        
    Returns:
        results: List of (path, predicted_class, confidence) tuples
    """
    results = []
    
    print(f"\n🔍 Classifying {len(image_paths)} images...\n")
    
    for path in image_paths:
        pred_class, conf, _ = predict_image(model, path, show=False)
        results.append((path, pred_class, conf))
        print(f"  {path}: {pred_class} ({conf:.1f}%)")
    
    return results


# =============================================================================
# Run inference directly
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict image class using trained CNN")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the image to classify")
    parser.add_argument("--model", type=str, default=BEST_MODEL_PATH,
                        help=f"Path to model checkpoint (default: {BEST_MODEL_PATH})")
    args = parser.parse_args()
    
    # Load model
    model = load_trained_model(args.model)
    
    # Predict
    pred_class, confidence, all_probs = predict_image(model, args.image)
    
    print(f"\n📊 Prediction: {pred_class} ({confidence:.1f}%)")
    print(f"\n📋 All probabilities:")
    for cls, prob in all_probs.items():
        bar = "█" * int(prob / 2)
        print(f"   {cls:12s}: {prob:5.1f}% {bar}")
