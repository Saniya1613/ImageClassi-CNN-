import os
import torch
import torch.nn.functional as F
import gradio as gr
from PIL import Image
from torchvision import transforms

# Import project utilities
from src.config import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, IMAGE_SIZE
from src.model import SimpleCNN

# Use CPU for deployment compatibility
device = torch.device("cpu")

# Path to the best trained model checkpoint
MODEL_PATH = "models/best_model.pt"

# Initialize model
model = None
model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        model = SimpleCNN().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"✅ Loaded model successfully from {MODEL_PATH} (Accuracy: {checkpoint.get('val_acc', 0.0):.2f}%)")
    except Exception as e:
        model_load_error = f"Error loading model weights: {str(e)}"
else:
    model_load_error = f"Model checkpoint not found at '{MODEL_PATH}'. Please train the model first using: `python main.py --mode train`"

# Input image preprocessing transform (must match training test transforms)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

def predict(img):
    if model_load_error:
        return {"Error": 1.0}, f"❌ Model Load Failure:\n{model_load_error}"
    
    if img is None:
        return None, "Please upload an image."

    try:
        # Preprocess PIL image
        img_rgb = img.convert("RGB")
        input_tensor = transform(img_rgb).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Map classes to float probabilities for Gradio Label
        results = {
            CIFAR10_CLASSES[i]: float(probabilities[i])
            for i in range(len(CIFAR10_CLASSES))
        }
        
        # Get top class prediction for status message
        top_idx = torch.argmax(probabilities).item()
        top_class = CIFAR10_CLASSES[top_idx]
        top_conf = probabilities[top_idx].item() * 100
        
        status_msg = f"✅ Classified as **{top_class.upper()}** with **{top_conf:.1f}%** confidence."
        return results, status_msg
        
    except Exception as e:
        return None, f"❌ Inference Error: {str(e)}"

# Define the custom CSS for styling the UI
custom_css = """
footer {visibility: hidden}
.gradio-container {
    font-family: 'Outfit', -apple-system, sans-serif !important;
}
"""

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald", secondary_hue="slate"), css=custom_css) as demo:
    gr.Markdown(
        """
        # 🧠 Custom CNN Image Classifier (CIFAR-10)
        
        Upload any image and the Convolutional Neural Network will classify it into one of the 10 CIFAR-10 classes:
        **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, or truck**.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Image")
            submit_btn = gr.Button("Classify Image", variant="primary")
            
        with gr.Column(scale=1):
            status_output = gr.Markdown(value="Upload an image and click classify to see results.")
            label_output = gr.Label(num_top_classes=5, label="Prediction Confidence")
            
    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[label_output, status_output]
    )
    
    gr.Markdown(
        """
        ---
        ### 📊 Project Highlights
        - **Custom Architecture**: 3-block Convolutional Neural Network trained from scratch.
        - **Dataset**: CIFAR-10 (60,000 32x32 color images).
        - **Framework**: PyTorch.
        """
    )

if __name__ == "__main__":
    demo.launch()
