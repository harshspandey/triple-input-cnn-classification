# 🎵 Triple-Input CNN Classification: Custom Deep Learning from Scratch

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![F1 Score](https://img.shields.io/badge/F1--Score-95%25-brightgreen)

## 📌 Overview
This repository contains a high-performance **Multi-Stream Convolutional Neural Network** designed to classify 16 musical instruments from **triple-spectrogram inputs** (three temporal views of the same sound).

**The Engineering Challenge:**
To demonstrate a first-principles understanding of Deep Learning, this model was architected **without using standard PyTorch layers** (`nn.Conv2d`, `nn.Linear`, `nn.BatchNorm2d`). 

Instead, I implemented the core mathematical operations (Convolution via `im2col`, Batch Normalization, and Max Pooling) using raw tensor math.

## 🧠 Model Architecture: "The Diamond"



The model uses a **Siamese-style architecture** with three identical branches that share weights to extract features from the three input views simultaneously.

### Key Technical Innovations:
1.  **Manual Convolution:** Implemented using `F.unfold` and matrix multiplication to reconstruct the convolution operation from scratch.
2.  **Mish Activation:** Replaced standard ReLU with a manual implementation of **Mish** ($x \cdot \tanh(\ln(1+e^x))$) for smoother gradient flow.
3.  **High-Performance Training:** Optimized for Dual-T4 GPUs using **Mixed Precision (AMP)** and `OneCycleLR` scheduling.
4.  **Handling Imbalance:** Solved a 3:1 class imbalance using **Class-Weighted Cross-Entropy Loss**, achieving >90% recall on rare classes.

## 📊 Results

The model achieves state-of-the-art performance on the validation dataset.

| Metric | Score |
|:-------|:------|
| **Accuracy** | **94.8%** |
| **F1-Score (Macro)** | **0.95** |

### Confusion Matrix
The model demonstrates excellent separation of classes, with a nearly perfect diagonal.

![Confusion Matrix](assets/confusion_matrix.png)

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/triple-input-cnn-classification.git](https://github.com/YOUR_USERNAME/triple-input-cnn-classification.git)
    cd triple-input-cnn-classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Notebook:**
    Open `Triple_Input_CNN.ipynb` in Jupyter Lab or Google Colab and execute all cells.

## 📂 Repository Structure
* `Triple_Input_CNN.ipynb`: The complete training, validation, and inference pipeline.
* `outputs/`: Contains the generated submission files and detailed classification metrics.
* `assets/`: Visualization of model performance.

## 📜 License
MIT License
