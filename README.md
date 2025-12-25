# Kidney Cancer Detection & Grading using CNN

A Deep Learning project that utilizes Convolutional Neural Networks (CNN) to detect kidney cancer from medical images and categorize them into histological grades. This project covers both binary classification (Normal vs. Tumor) and multi-class cancer grading.

## 🚀 Overview

Early detection of kidney cancer is crucial for successful treatment. This repository contains two implementations:

1.  **Binary Classification**: Identifying if a kidney image is 'Normal' or 'Tumor'.
2.  **Histological Grading**: Classifying tumor images into one of five grades (Grade 0 to Grade 4) based on severity.

## ✨ Key Features

- **Data Automation**: Automated dataset download and extraction from Google Drive using `gdown` and `rarfile`.
- **Image Preprocessing**: Dynamic resizing (224x224) and normalization using `ImageDataGenerator`.
- **Deep CNN Architecture**: Multi-layered Convolutional Neural Network built with Keras Sequential API.
- **Performance Visualization**: Real-time plotting of training/validation loss and accuracy.
- **Evaluation**: Final accuracy metrics on unseen test data.

## 📊 Dataset

The project uses histological images of kidney tissue.

- **Binary Dataset**: Contains `kidney_normal` and `kidney_tumor` classes.
- **Grading Dataset**: Contains five classes: `Grade0`, `Grade1`, `Grade2`, `Grade3`, and `Grade4`.

## 🏗️ Model Architecture

The models share a robust architectural foundation:

- **Convolutional Layers**: 5 layers with increasing filters (32 -> 64 -> 128 -> 256 -> 512).
- **Activation**: ReLU for non-linearity.
- **Pooling**: Max Pooling with 2x2 strides for spatial reduction.
- **Fully Connected**: High-density hidden layers (512 and 1024 nodes).
- **Output Layer**:
  - Softmax with 2 nodes for Binary Classification.
  - Softmax with 5 nodes for Grading Classification.

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.8+
- TensorFlow / Keras
- Google Colab (Recommended) or Local Jupyter Environment

### Execution

1. Open the `.ipynb` files in Google Colab or Jupyter.
2. Ensure you have internet access (for dataset download).
3. Run the cells sequentially to:
   - Download and extract the dataset.
   - Preprocess and augment image data.
   - Construct the CNN model.
   - Train the model (default 10 epochs).
   - Evaluate and view plots.

## 📈 Results

| Implementation        | Accuracy   | Final Loss |
| :-------------------- | :--------- | :--------- |
| Binary (Normal/Tumor) | **99.95%** | 0.0049     |
| Grading (5 Classes)   | **27.08%** | 1.5970     |

_Note: The grading model accuracy reflects initial training on a specific subset; further fine-tuning or more data can improve these results._

## 💻 Technologies Used

- **Language**: Python
- **Deep Learning**: TensorFlow, Keras
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Utils**: Gdown, Rarfile, PIL
