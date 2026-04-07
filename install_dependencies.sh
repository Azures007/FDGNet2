#!/bin/bash

# FGANet Environment Setup Script
# FGANet 环境一键安装脚本

echo "========================================="
echo "Starting FGANet environment setup..."
echo "开始配置 FGANet 运行环境..."
echo "========================================="

# 1. Update pip to the latest version
# 1. 更新 pip 到最新版本
echo "Step 1/3: Updating pip..."
pip install --upgrade pip

# 2. Install PyTorch dependencies (Assuming PyTorch is already installed in the base image)
# If not, uncomment the following line (select version based on your CUDA version):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install project-specific dependencies
# 3. 安装项目所需的 Python 库
echo "Step 2/3: Installing required libraries..."

# Basic data processing and scientific computing
# 基础数据处理和科学计算
pip install numpy pandas scipy scikit-learn h5py

# Image and visualization
# 图像处理和可视化
pip install imageio matplotlib

# Hyperspectral data processing
# 高光谱数据处理
pip install spectral

# Deep learning models and utilities
# 深度学习模型库
pip install timm

echo "Step 3/3: Verifying installation..."
python -c "import torch; import imageio; import timm; import h5py; import pandas; import spectral; print('All dependencies installed successfully!')"

echo "========================================="
echo "Setup completed! You can now run the training script."
echo "环境配置完成！现在可以运行训练脚本了。"
echo "========================================="
