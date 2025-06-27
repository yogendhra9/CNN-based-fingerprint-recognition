# Fingerprint Classification using ResNet18

A deep learning project for classifying fingerprint types using a modified ResNet18 architecture with PyTorch. The model achieves over 97% accuracy with custom enhancements and test-time augmentation.

---

## 📌 Project Overview

This project implements a robust fingerprint classification pipeline using:

* Pre-trained ResNet18 for feature extraction
* Advanced fingerprint enhancement techniques
* Strong data augmentation for generalization
* Focal Loss for hard example emphasis
* Cosine Annealing learning rate scheduler
* Test-Time Augmentation (TTA) for improved inference

Achieved 97.17% accuracy after 20 epochs on a real fingerprint dataset.

---

## 🧠 Motivation

Traditional CNNs like LeNet achieved modest results (\~69%). This project aims to significantly boost accuracy using a deeper ResNet-based architecture, stronger preprocessing, and regularization techniques.

---

## 📂 Dataset

We use a publicly available fingerprint image dataset from Kaggle:

**🔗 Dataset**: [SOCOFing - Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)

* **Total Images**: 6000+
* **Classes**: 10 (e.g., Left thumb, Right index, etc.)
* **Format**: JPEG images of 96x103 pixels

Upload or unzip the dataset into:

```bash
/content/fingerprint_images/Real
```

Example filename: `0001_Left_index_finger_BW.png`

---

## 🧱 Project Structure

```bash
Fingerprint-Classifier/
|
├── enhanced_fingerprint_model.pth      # Saved PyTorch model
├── training_curves.png                 # Loss and accuracy plot
├── fingerprint_classifier.py           # Full training and evaluation code
├── README.md                           # Project documentation
└── requirements.txt                    # Dependencies
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Fingerprint-Classifier.git
cd Fingerprint-Classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision matplotlib numpy opencv-python scikit-learn tqdm
```

### 3. Prepare dataset

Download and extract the SOCOFing dataset from Kaggle:
[https://www.kaggle.com/datasets/ruizgara/socofing](https://www.kaggle.com/datasets/ruizgara/socofing)

Place it under:

```bash
/content/fingerprint_images/Real
```

### 4. Run the model

```bash
python fingerprint_classifier.py
```

Trains and saves model → `enhanced_fingerprint_model.pth`
Final accuracy: **97.17%** on validation set with TTA.

---

## 📊 Model Architecture

Modified ResNet18 (pretrained on ImageNet):

* 18 convolutional layers

* Final FC layer replaced for 10-class classification

* Dropout + BatchNorm added to avoid overfitting

* 🧠 Loss: FocalLoss

* ⚙️ Optimizer: AdamW

* 📉 LR Scheduler: CosineAnnealingWarmRestarts

---

## 📊 Results

| Epochs | Accuracy |
| ------ | -------- |
| 6      | 78.33%   |
| 15     | 87.45%   |
| 20     | 97.17% ✅ |

Enhanced input processing and augmentation significantly improved results.

---


## 📌 Highlights

* ✅ Strong preprocessing (CLAHE, bilateral filtering, thresholding)
* ✅ Test-time augmentation for more robust predictions
* ✅ Clean and modular PyTorch code
* ✅ Easily reproducible for custom datasets

---

## 💡 Future Work

* Try ResNet50 / DenseNet for further accuracy gains
* Implement Grad-CAM for visual explanation
* Convert model to ONNX or TFLite for mobile inference

---

## 🤝 Credits

* **Dataset**: [SOCOFing – Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)
* **Base model**: torchvision.models.resnet18
* **Developed by**: \[Your Name Here]

---



---

Feel free to fork, improve, or use in your own projects — contributions welcome!
