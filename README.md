# 🧠 Image Classification with CNNs (PyTorch)

A complete, end-to-end image classification pipeline built with **Convolutional Neural Networks** using PyTorch. This project classifies images from the **CIFAR-10** dataset into 10 categories.

## 📸 Dataset: CIFAR-10

| Property | Value |
|---|---|
| **Classes** | airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck |
| **Images** | 60,000 (50k train / 10k test) |
| **Size** | 32×32 pixels, RGB color |
| **Source** | Auto-downloaded by PyTorch |

## 🏗️ Project Structure

```
IMageClass/
├── main.py                # Unified entry point
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters & constants
│   ├── dataset.py         # Data loading & transforms
│   ├── model.py           # Custom CNN architecture
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation & metrics
│   ├── visualize.py       # Plotting utilities
│   ├── predict.py         # Inference on custom images
│   └── transfer_learn.py  # Transfer learning (ResNet-18)
└── models/                # Saved model weights (gitignored)
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the Custom CNN
```bash
python main.py --mode train
```

### 3. Evaluate
```bash
python main.py --mode evaluate
```

### 4. Transfer Learning (ResNet-18)
```bash
python main.py --mode transfer
```

### 5. Predict on Your Own Image (CLI)
```bash
python main.py --mode predict --image path/to/your/image.jpg
```

### 6. Run the Interactive Web App (Gradio)
To run a local web interface for real-time image uploads and classification:
```bash
python app.py
```
Once started, open `http://127.0.0.1:7860` in your web browser.

## 🌐 Deploying to Hugging Face Spaces

This project is ready to be hosted for free on Hugging Face Spaces:

1. **Create a Space**:
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces) and click **Create new Space**.
   - Set the SDK to **Gradio** and select the free CPU basic tier.

2. **Commit Trained Model Weights**:
   - The model checkpoint `best_model.pt` is gitignored locally. You need to force-add it so it is included in your deployment:
     ```bash
     git add -f models/best_model.pt
     git commit -m "chore: include best_model.pt weights for deployment"
     ```

3. **Push to Hugging Face**:
   - Add your Hugging Face Space repository as a remote (replace with your username and Space name):
     ```bash
     git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
     git push -f hf main
     ```

## 🧪 What You'll Learn

- **Convolutional Neural Networks** — how convolutions detect features in images
- **Data Augmentation** — making your model robust with random transforms
- **Batch Normalization** — stabilizing and accelerating training
- **Dropout** — preventing overfitting
- **Transfer Learning** — leveraging pretrained models for better accuracy
- **Evaluation Metrics** — confusion matrices, precision, recall, F1 scores

## 🛠️ Tech Stack

- Python 3.10+
- PyTorch + torchvision
- matplotlib + seaborn
- scikit-learn
- NumPy

## 📊 Results

> Results will be added after training is complete.

## 📝 License

This project is for educational purposes.
