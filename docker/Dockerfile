# Start from the PyTorch image with CUDA 11.8
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-devel

# Install system packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential \
    libgl1 \
    udev \
    libsm6 \
    libxext6

# Python package installation
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    tqdm \
    einops \
    monai[all]

# If specific versions are not necessary, remove version pins for the latest compatible release
RUN pip install --no-cache-dir \
    matplotlib scipy seaborn pyyaml tab-transformer-pytorch

# Working directory for your application
WORKDIR /home/work

