#!/bin/bash
# run_training.sh

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Example 1: Basic 4-GPU training with default settings
echo "Starting 4-GPU training with default settings..."
python multi_gpu_train.py \
    --data_path ./yolo_dataset_2_5d_byu \
    --device 0,1,2,3 \
    --batch_size 64 \
    --workers 8 \
    --epochs 100

# Example 2: High-performance training with larger batch size
# python multi_gpu_train.py \
#     --data_path ./yolo_dataset_2_5d_byu \
#     --device 0,1,2,3 \
#     --batch_size 128 \
#     --workers 12 \
#     --epochs 150 \
#     --imgsz 960 \
#     --lr0 2e-4 \
#     --name high_performance_training

# Example 3: Memory-efficient training for limited VRAM
# python multi_gpu_train.py \
#     --data_path ./yolo_dataset_2_5d_byu \
#     --device 0,1,2,3 \
#     --batch_size 32 \
#     --workers 4 \
#     --epochs 200 \
#     --imgsz 640 \
#     --name memory_efficient_training

# Example 4: Custom augmentation settings
# python multi_gpu_train.py \
#     --data_path ./yolo_dataset_2_5d_byu \
#     --device 0,1,2,3 \
#     --batch_size 64 \
#     --workers 8 \
#     --epochs 100 \
#     --mosaic 0.8 \
#     --mixup 0.3 \
#     --degrees 30 \
#     --scale 0.2 \
#     --name custom_augmentation
