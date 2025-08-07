# 🌸 Flower Image Classifier - Udacity ML Nanodegree Project

**This Project was first submitted on September 11, 2024. Later on, it went through some enhancements that are listed below.**

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.1-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Conda](https://img.shields.io/badge/Conda-Environment-green?logo=anaconda)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

**Keywords**: `machine learning`, `deep learning`, `tensorflow`, `image classification`, `transfer learning`, `mobilenetv2`, `computer vision`, `neural networks`, `udacity nanodegree`, `flower classification`, `oxford flowers 102`, `conda environment`, `jupyter notebook`, `python`, `artificial intelligence`

## 📋 Project Overview

This repository contains a **complete solution** to the **Introduction to Machine Learning with TensorFlow Nanodegree** Image Classifier Project. The project implements a deep learning model using transfer learning to classify 102 different flower species with high accuracy.

## 🎯 Project Objectives

- Build an image classifier using TensorFlow and pre-trained neural networks
- Implement transfer learning with MobileNetV2 architecture
- Create a robust data preprocessing pipeline
- Develop inference functions for real-world image classification
- Achieve >70% accuracy on unseen test data

## 🚀 Key Contributions & Improvements

### ✨ **Enhanced Architecture**
- **Replaced TensorFlow Hub with Keras Applications**: Solved compatibility issues by using `tf.keras.applications.MobileNetV2` instead of TensorFlow Hub
- **Improved Model Structure**: Added GlobalAveragePooling2D and Dropout layers for better performance and regularization
- **Functional API Implementation**: Used TensorFlow's Functional API for more stable model creation

### 🔧 **Development Environment Setup**
- **Conda Virtual Environment**: Created isolated environment `flower-classifier` with Python 3.9
- **Dependency Management**: Properly managed TensorFlow, TensorFlow Datasets, and all required packages
- **Cross-Platform Compatibility**: Enhanced GPU detection for macOS (Intel & Apple Silicon) and other platforms

### 🖥️ **System Optimization**
- **Apple Silicon Support**: Added Metal Performance Shaders (MPS) detection for Apple Silicon Macs
- **Smart GPU Detection**: Implemented platform-aware GPU detection and memory management
- **Comprehensive System Info**: Enhanced system information display for better debugging

### 📊 **Code Quality Improvements**
- **Error Handling**: Added robust error handling throughout the codebase
- **Code Documentation**: Comprehensive comments and documentation
- **Modular Design**: Clean separation of concerns in data processing, model building, and inference

## 🛠️ Technical Stack

- **Framework**: TensorFlow 2.18.1
- **Architecture**: MobileNetV2 (Transfer Learning)
- **Dataset**: Oxford Flowers 102 (via TensorFlow Datasets)
- **Environment**: Conda Virtual Environment
- **Platform**: Cross-platform (macOS, Linux, Windows)

## 📁 Project Structure

```
image_classifier/
├── Project_Image_Classifier_Project_Solution.ipynb  # Main notebook complete solution
├── predict.py                                       # Command-line prediction script
├── setup_environment.sh                             # Automated environment setup script
├── label_map.json                                   # Flower class mappings
├── README.md                                        # This file
├── .gitignore                                       # Git ignore rules for Python/ML projects
├── *.keras                                          # Trained model files (generated after training)
├── assets/                                          # Project images and resources
│   ├── Flowers.png
│   └── inference_example.png
└── test_images/                                     # Sample images for testing
    ├── cautleya_spicata.jpg
    ├── hard-leaved_pocket_orchid.jpg
    ├── orange_dahlia.jpg
    └── wild_pansy.jpg
```

## 🏗️ Setup Instructions

### Quick Setup (Recommended)
Run the automated setup script:
```bash
chmod +x setup_environment.sh
./setup_environment.sh
```

### Manual Setup (Alternative)

### 1. Create Conda Environment
```bash
conda create -n flower-classifier python=3.9 -y
conda activate flower-classifier
```

### 2. Install Dependencies
```bash
conda install tensorflow matplotlib numpy pillow jupyter ipykernel -c conda-forge -y
pip install tensorflow-datasets
```

### 3. Register Jupyter Kernel
```bash
python -m ipykernel install --user --name flower-classifier --display-name "Python (flower-classifier)"
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

### 5. Verify Installation
```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

## 🎯 Model Performance

- **Training Accuracy**: ~95%+
- **Validation Accuracy**: ~85%+
- **Test Accuracy**: ~80%+ (exceeds 70% requirement)
- **Model Size**: Optimized for deployment
- **Inference Time**: Fast prediction on single images

## 💡 Key Features

### 🔍 **Data Pipeline**
- Automatic dataset download and preprocessing
- Image normalization and resizing to 224x224
- Efficient batching and prefetching
- Data augmentation ready

### 🧠 **Model Architecture**
```python
Model: MobileNetV2 + Custom Classifier
├── Input Layer (224, 224, 3)
├── MobileNetV2 Base (frozen)
├── GlobalAveragePooling2D
├── Dropout(0.2)
└── Dense(102, softmax)
```

### 📈 **Training Strategy**
- Transfer learning with frozen base model
- Adam optimizer with sparse categorical crossentropy
- 10 epochs training with validation monitoring
- Early stopping capability

### 🔮 **Inference Pipeline**
- Image preprocessing function
- Top-K predictions with probabilities
- Visual prediction results with matplotlib
- Command-line interface for batch processing

## 📊 Results & Visualizations

The notebook includes comprehensive visualizations:
- Training/validation accuracy and loss curves
- Sample predictions with confidence scores
- Confusion matrix analysis
- Model architecture summary

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Deep Learning**: Transfer learning, fine-tuning, model evaluation
- **TensorFlow**: Model building, training, saving/loading
- **Data Science**: Data preprocessing, visualization, analysis
- **Software Engineering**: Environment management, code organization
- **Problem Solving**: Debugging compatibility issues, optimization

## 🌟 Beyond the Requirements

This solution goes beyond the basic project requirements by:
- ✅ **Automated Setup Script**: One-command environment setup with `setup_environment.sh`
- ✅ Enhanced error handling and debugging capabilities
- ✅ Cross-platform compatibility (macOS/Linux/Windows)
- ✅ Modern TensorFlow best practices
- ✅ Comprehensive documentation and comments
- ✅ Production-ready code structure
- ✅ Conda environment for reproducibility

## 📝 Usage Examples

### Jupyter Notebook
1. Run the setup script: `./setup_environment.sh`
2. Activate environment: `conda activate flower-classifier`
3. Open `Project_Image_Classifier_Project.ipynb`
4. Select "Python (flower-classifier)" kernel
5. Run all cells sequentially

### Command Line
```bash
python predict.py ./test_images/wild_pansy.jpg ./model.keras --top_k 5
```

## 🤝 Acknowledgments

- **Udacity**: For the excellent Machine Learning Nanodegree program
- **TensorFlow Team**: For the robust deep learning framework
- **Oxford VGG Group**: For the Flowers 102 dataset

## 📄 License

This project is part of the Udacity Machine Learning Nanodegree program and is intended for educational purposes.

---

**Author**: Amr Ramadan  
**Program**: Udacity Introduction to Machine Learning with TensorFlow Nanodegree  
**Project**: Image Classifier  
**Date**: 2025
