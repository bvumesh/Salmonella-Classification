# 🦠 Salmonella Image Classification

This repository presents a deep learning and hybrid machine learning framework for classifying poultry fecal images as **Salmonella-Positive** or **Negative**. It includes training, evaluation, and feature-based classification pipelines using state-of-the-art convolutional neural networks (CNNs) and traditional ML models.

---

## 📁 Repository Structure

| File Name                              | Description |
|----------------------------------------|-------------|
| `Massive_Model_Training.ipynb`         | Trains CNN models from scratch and saves `.h5` Keras models. |
| `Fine_tuned_Training_to_Save_PTH_Files.ipynb` | Fine-tunes models and saves `.pth` PyTorch models. |
| `Hybrid_DL_ML_Classifier.ipynb`        | Extracts deep features from CNNs and uses them in traditional ML models. |
| `Metrics_using_H5_files.ipynb`         | Evaluates `.h5` models using various metrics (Accuracy, ROC, PR curves, etc.). |
| `Metrics_using_PTH_files.ipynb`        | Evaluates `.pth` models (from PyTorch) on test data. |

---

## 🧠 Models Included in Massive Model Training

Both **with and without fine-tuning**:

- VGG16
- VGG19
- MobileNetV2 / V3
- DenseNet121 / 169 / 201
- Xception
- InceptionV3
- InceptionResNetV2
- InceptionV4 (*via keras-applications*)
- ResNet50 / 101 / 152
- EfficientNetB0 / B3 / B7
- NASNetMobile / NASNetLarge

Each model undergoes:
1. **Initial training with frozen base layers**
2. **Fine-tuning (unfreezing top layers)** for better performance

---

## 🔬 Hybrid DL-ML Classifier

The `Hybrid_DL_ML_Classifier.ipynb` performs:
- Feature extraction using CNN backbones (e.g., VGG16, DenseNet)
- Classification using ML models:
  - Random Forest
  - SVM
  - Logistic Regression
  - k-Nearest Neighbors
  - Gradient Boosting

**Metrics evaluated:**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC / PR Curves

---

## 🛠️ Installation

Install the required dependencies:

```bash
# Deep Learning Libraries
pip install tensorflow
pip install keras-applications  # for InceptionV4
pip install torch torchvision   # for PyTorch models

# Data Science Utilities
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python

# Optional: For augmentations
pip install albumentations
```

---

## 📂 Dataset Structure

Organize your image dataset as follows:

```bash
/dataset/
  ├── train/
  │   ├── healthy/
  │   └── salmo/
  ├── valid/
  │   ├── healthy/
  │   └── salmo/
  └── test/
      ├── healthy/
      └── salmo/
```
📌 Update all path variables in the notebooks to match your dataset location.

---

## 📈 Outputs Generated

- Model training curves (accuracy/loss)
- Saved model weights (.h5, .pth)
- ROC and PR curves
- Confusion matrices
- Classification reports
- Exported Excel summary (optional)

---

