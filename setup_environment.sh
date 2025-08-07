#!/bin/bash

# Flower Image Classifier - Environment Setup Script
# This script creates and configures the conda environment for the project

set -e  # Exit on any error

echo "🌸 Setting up Flower Image Classifier Environment..."
echo "=================================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first"
    exit 1
fi

# Environment name
ENV_NAME="flower-classifier"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        echo "❌ Setup cancelled"
        exit 1
    fi
fi

echo "🔧 Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.9 -y

echo "📦 Installing dependencies..."
# Activate environment and install packages
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install packages via conda (faster and more reliable for these packages)
conda install tensorflow matplotlib numpy pillow jupyter ipykernel -c conda-forge -y

# Install tensorflow-datasets via pip (not available in conda-forge)
pip install tensorflow-datasets

echo "🎯 Registering Jupyter kernel..."
python -m ipykernel install --user --name $ENV_NAME --display-name "Python (flower-classifier)"

echo "✅ Environment setup complete!"
echo ""
echo "🚀 To use the environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "📓 To launch Jupyter Notebook:"
echo "   jupyter notebook Project_Image_Classifier_Project.ipynb"
echo ""
echo "🔍 To verify installation:"
echo "   python -c \"import tensorflow as tf; print('TensorFlow version:', tf.__version__)\""
