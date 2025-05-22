# 🧠 English Digit Classifier (MNIST) with CNN

A deep learning project using Convolutional Neural Networks (CNNs) to classify handwritten English digits (0–9) from the MNIST dataset. The model is trained and evaluated using TensorFlow/Keras and includes data augmentation for improved generalization.

---

## 📌 Project Overview

This project demonstrates how to build, train, and evaluate a CNN on the MNIST dataset of handwritten digits. It also includes support for data augmentation to enhance training robustness.

---

## 🧰 Tech Stack

- Python 3.10+
- TensorFlow / Keras
- NumPy, OpenCV
- Matplotlib

---

## 🗂️ Directory Structure

📁 digit-classifier/
│
├── model_english.keras # Trained CNN model for English digits
├── main.py # Main training script
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## 🧪 Dataset

- **MNIST** — 60,000 training and 10,000 test grayscale digit images (28x28 pixels).
- Dataset is automatically downloaded via Keras.

---

## 🚀 How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/PIYUSHMISHRA1009/-Ultra-Advanced-Digit-Classification.git
   cd english-digit-classifier
2. **Create a virtual environment (optional but recommended)**
    python3 -m venv .venv
    source .venv/bin/activate

3. **Install dependencies**

    pip install -r requirements.txt
   
4. **Train the model**

    python main.py

**The trained model is saved as:**
model_english.keras

**Results**
The model achieves ~99% accuracy on the MNIST test set after training for 15 epochs with data augmentation.

