"""
model.py — Custom CNN Architecture for CIFAR-10
================================================

This is the HEART of the project. Here we define a Convolutional Neural Network
from scratch using PyTorch's nn.Module.

╔══════════════════════════════════════════════════════════════════════╗
║                    WHAT IS A CNN?                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  A CNN is a type of neural network designed specifically for images.  ║
║  It uses CONVOLUTION operations to automatically learn features       ║
║  like edges, textures, shapes, and objects.                           ║
║                                                                      ║
║  Think of it like this:                                               ║
║  - Layer 1: Detects simple features (edges, corners)                 ║
║  - Layer 2: Combines those into textures (fur, feathers)             ║
║  - Layer 3: Combines textures into parts (ears, wheels)              ║
║  - Final layers: Uses those parts to classify (cat? car? plane?)     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

KEY BUILDING BLOCKS:

1. CONV2D (Convolutional Layer)
   - Slides small filters (e.g., 3×3) across the image
   - Each filter detects a specific feature (edge, color pattern, etc.)
   - Output: "feature maps" — one per filter
   - Parameters: input_channels, output_channels, kernel_size

2. BATCHNORM (Batch Normalization)
   - Normalizes outputs of each layer to have mean=0, std=1
   - WHY: Prevents "internal covariate shift" — as weights update,
     the distribution of each layer's inputs keeps changing, making
     training unstable. BatchNorm fixes this.
   - Bonus: Acts as mild regularization

3. RELU (Activation Function)
   - ReLU(x) = max(0, x) — keeps positive values, zeros out negatives
   - WHY: Without activation functions, stacking layers would just be
     one big linear transformation (useless!). ReLU adds non-linearity
     so the network can learn complex patterns.

4. MAXPOOL (Max Pooling)
   - Downsamples feature maps by taking the max in each 2×2 window
   - WHY: Reduces spatial dimensions (32→16→8→4), which:
     a) Reduces computation
     b) Gives the network "translation invariance" (a cat in the
        top-left looks the same as a cat in the bottom-right)

5. DROPOUT
   - Randomly sets a fraction of neurons to 0 during training
   - WHY: Forces the network to not rely on any single neuron.
     This prevents OVERFITTING (memorizing training data instead
     of learning general patterns). Like studying with random
     flashcards removed — you learn the concepts, not the cards.

6. FULLY CONNECTED (Linear Layer)
   - Traditional neural network layer — every input connects to every output
   - Used at the end to map learned features → class predictions
"""

import torch
import torch.nn as nn

from src.config import NUM_CLASSES, NUM_CHANNELS


class SimpleCNN(nn.Module):
    """
    A 3-block Convolutional Neural Network for CIFAR-10 classification.
    
    Architecture:
    ┌──────────────────────────────────────────────────────┐
    │ INPUT: 3×32×32 (RGB image)                           │
    ├──────────────────────────────────────────────────────┤
    │ Block 1: Conv(3→32) → BatchNorm → ReLU → MaxPool    │
    │          Output: 32×16×16                             │
    ├──────────────────────────────────────────────────────┤
    │ Block 2: Conv(32→64) → BatchNorm → ReLU → MaxPool   │
    │          Output: 64×8×8                               │
    ├──────────────────────────────────────────────────────┤
    │ Block 3: Conv(64→128) → BatchNorm → ReLU → MaxPool  │
    │          Output: 128×4×4                              │
    ├──────────────────────────────────────────────────────┤
    │ Flatten: 128×4×4 = 2048                              │
    ├──────────────────────────────────────────────────────┤
    │ FC1: 2048 → 256 → ReLU → Dropout(0.5)               │
    ├──────────────────────────────────────────────────────┤
    │ FC2: 256 → 10 (one output per class)                 │
    └──────────────────────────────────────────────────────┘
    
    Total parameters: ~600K (small enough to train on CPU in ~20 min)
    """
    
    def __init__(self, num_classes=NUM_CLASSES):
        """
        Initialize all layers of the CNN.
        
        This is where we DEFINE the architecture. The layers are not
        connected yet — that happens in forward().
        """
        super(SimpleCNN, self).__init__()
        
        # =====================================================================
        # FEATURE EXTRACTOR — Convolutional blocks that learn image features
        # =====================================================================
        # We use nn.Sequential to chain layers together cleanly.
        # Data flows through them in order: Conv → BN → ReLU → Pool
        
        # Block 1: 3 channels → 32 feature maps
        # Input: (batch, 3, 32, 32) → Output: (batch, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=NUM_CHANNELS,  # 3 (RGB)
                out_channels=32,           # Learn 32 different filters
                kernel_size=3,             # Each filter is 3×3 pixels
                padding=1,                 # Pad input so output has same H×W
            ),
            nn.BatchNorm2d(32),           # Normalize 32 feature maps
            nn.ReLU(inplace=True),        # Activation (inplace saves memory)
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32×32 → 16×16
        )
        
        # Block 2: 32 → 64 feature maps
        # Input: (batch, 32, 16, 16) → Output: (batch, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16×16 → 8×8
        )
        
        # Block 3: 64 → 128 feature maps
        # Input: (batch, 64, 8, 8) → Output: (batch, 128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8×8 → 4×4
        )
        
        # =====================================================================
        # CLASSIFIER — Fully connected layers that map features → predictions
        # =====================================================================
        # After 3 pooling layers: 32→16→8→4, with 128 channels = 128×4×4 = 2048
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (batch, 128, 4, 4) → (batch, 2048)
            nn.Linear(128 * 4 * 4, 256),  # 2048 → 256 neurons
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),            # Kill 50% of neurons randomly
            nn.Linear(256, num_classes),  # 256 → 10 class scores
        )
    
    def forward(self, x):
        """
        Define how data flows through the network.
        
        This is called automatically when you do: output = model(images)
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Tensor of shape (batch_size, 10) — raw scores for each class
            (these are called "logits" — NOT probabilities yet!)
            
            To get probabilities, apply softmax: probs = F.softmax(output, dim=1)
            But for training, CrossEntropyLoss does this internally.
        """
        # Pass through convolutional blocks
        x = self.block1(x)   # (batch, 3, 32, 32) → (batch, 32, 16, 16)
        x = self.block2(x)   # (batch, 32, 16, 16) → (batch, 64, 8, 8)
        x = self.block3(x)   # (batch, 64, 8, 8) → (batch, 128, 4, 4)
        
        # Pass through classifier
        x = self.classifier(x)  # (batch, 128, 4, 4) → (batch, 10)
        
        return x


def model_summary(model):
    """
    Print a summary of the model architecture and parameter count.
    
    This helps you understand:
    - How many parameters (weights) the model has
    - Which layers are trainable
    - The relative size of each component
    """
    print("=" * 60)
    print("  MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(model)
    print("=" * 60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Count:")
    print(f"   Total:     {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Size:      ~{total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    
    # Breakdown by block
    print(f"\n📦 Per-Block Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"   {name:15s}: {params:>8,} parameters")
    
    print("=" * 60)
    return total_params


# =============================================================================
# Test the model
# =============================================================================
if __name__ == "__main__":
    # Create the model
    model = SimpleCNN()
    
    # Print summary
    model_summary(model)
    
    # Test with a dummy input (simulates one batch of 4 images)
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    
    print(f"\n🧪 Test Forward Pass:")
    print(f"   Input shape:  {dummy_input.shape}  (batch=4, C=3, H=32, W=32)")
    print(f"   Output shape: {output.shape}  (batch=4, classes=10)")
    print(f"   Output[0]:    {output[0].detach().numpy().round(3)}")
    print(f"   Predicted:    {output.argmax(dim=1).tolist()}")
