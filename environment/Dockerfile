# Use official NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /user/source

# Create necessary directories
RUN mkdir -p /user/source/data/images_HnE/train \
    /user/source/data/masks/train \
    /user/source/data/images_HnE/test \
    /user/source/data/masks/test \
    /user/source/data/images_IHC \
    /user/source/data/masks_IHC \
    /user/source/data/masks_IHC_cells \
    /user/source/models \
    /user/source/environment

# Copy scripts and requirements
COPY scripts/ /user/source/scripts/
COPY data/ /user/source/data/
COPY environment/requirements.txt /user/source/environment/
COPY environment/.env /user/source/environment/

# Install Git and dependencies
RUN apt-get update && apt-get install -y git \
    && apt-get install -y --no-install-recommends python3-pip ffmpeg libsm6 libxext6 \
    && pip3 install --upgrade pip \
    && pip3 install -r /user/source/environment/requirements.txt \
    && pip3 install efficientnet --upgrade \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Default environment variable to choose mode (can be overridden at runtime)
ENV RUN_MODE="training"

# Entrypoint script to switch between modes
COPY entrypoint.sh /user/source/entrypoint.sh
RUN chmod +x /user/source/entrypoint.sh

ENTRYPOINT ["/user/source/entrypoint.sh"]