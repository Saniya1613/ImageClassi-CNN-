"""
main.py — Unified Entry Point for the CNN Image Classifier
===========================================================

This is the single command you run for EVERYTHING:

    python main.py --mode train        # Train custom CNN from scratch
    python main.py --mode evaluate     # Evaluate saved model
    python main.py --mode transfer     # Transfer learning with ResNet-18
    python main.py --mode predict --image path/to/image.jpg  # Classify any image

All modes share the same underlying modules. This file just ties them together
with a clean CLI interface.
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="🧠 CNN Image Classifier — CIFAR-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode train                          # Train custom CNN (20 epochs)
  python main.py --mode train --epochs 5               # Quick training (5 epochs)
  python main.py --mode evaluate                       # Evaluate best model
  python main.py --mode transfer                       # Fine-tune ResNet-18
  python main.py --mode transfer --transfer-mode feature_extract
  python main.py --mode predict --image cat.jpg        # Classify an image
        """,
    )
    
    parser.add_argument(
        "--mode", type=str, required=True,
        choices=["train", "evaluate", "predict", "transfer"],
        help="What to do: train, evaluate, predict, or transfer",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (default: 20 for train, 10 for transfer)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image for prediction (required for --mode predict)",
    )
    parser.add_argument(
        "--transfer-mode", type=str, default="fine_tune",
        choices=["feature_extract", "fine_tune"],
        help="Transfer learning mode (default: fine_tune)",
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # MODE: TRAIN — Train the custom CNN from scratch
    # =========================================================================
    if args.mode == "train":
        from src.train import train_model
        from src.visualize import plot_training_curves
        from src.config import NUM_EPOCHS, LEARNING_RATE
        
        epochs = args.epochs or NUM_EPOCHS
        lr = args.lr or LEARNING_RATE
        
        model, history = train_model(num_epochs=epochs, learning_rate=lr)
        
        # Plot training curves after training
        print("\n📊 Generating training curves...")
        plot_training_curves(history)
    
    # =========================================================================
    # MODE: EVALUATE — Load best model and run full evaluation
    # =========================================================================
    elif args.mode == "evaluate":
        from src.evaluate import load_and_evaluate
        from src.visualize import show_predictions, show_misclassified
        from src.dataset import get_test_loader
        from src.config import DEVICE
        
        model, y_pred, y_true, y_probs = load_and_evaluate()
        
        # Also show visual predictions
        print("\n📊 Generating prediction visualizations...")
        test_loader = get_test_loader()
        show_predictions(model, test_loader, DEVICE)
        show_misclassified(model, test_loader, DEVICE)
    
    # =========================================================================
    # MODE: TRANSFER — Transfer learning with ResNet-18
    # =========================================================================
    elif args.mode == "transfer":
        from src.transfer_learn import train_transfer_model
        from src.visualize import plot_training_curves
        
        epochs = args.epochs or 10
        
        model, history = train_transfer_model(
            mode=args.transfer_mode,
            num_epochs=epochs,
        )
        
        print("\n📊 Generating training curves...")
        plot_training_curves(history, save_path="transfer_training_curves.png")
    
    # =========================================================================
    # MODE: PREDICT — Classify a custom image
    # =========================================================================
    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode")
        
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            sys.exit(1)
        
        from src.predict import load_trained_model, predict_image
        
        model = load_trained_model()
        pred_class, confidence, all_probs = predict_image(model, args.image)
        
        print(f"\n{'=' * 40}")
        print(f"  🏷️  Prediction: {pred_class.upper()}")
        print(f"  📊 Confidence: {confidence:.1f}%")
        print(f"{'=' * 40}")
        print(f"\n📋 All class probabilities:")
        for cls, prob in all_probs.items():
            bar = "█" * int(prob / 2)
            print(f"   {cls:12s}: {prob:5.1f}% {bar}")


if __name__ == "__main__":
    main()
