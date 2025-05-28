# multi_gpu_train.py
"""
Multi-GPU YOLO11m Training Script for Motor Detection
Supports DDP training with configurable batch size and workers
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
import random
import zipfile

def setup_args():
    """Setup command line arguments for flexible training configuration"""
    parser = argparse.ArgumentParser(description='Multi-GPU YOLO11m Motor Detection Training')
    
    # Training parameters
    parser.add_argument('--data_path', type=str, default='./yolo_dataset_2_5d_byu', 
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Total batch size for all GPUs (will be divided by number of GPUs)')
    parser.add_argument('--imgsz', type=int, default=960, 
                       help='Input image size')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of data loader workers per GPU')
    parser.add_argument('--device', type=str, default='0,1,2,3', 
                       help='GPU device(s) to use for training (e.g., "0,1,2,3" for 4 GPUs)')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='yolo11m.pt', 
                       help='Model weights path')
    parser.add_argument('--project', type=str, default='./runs', 
                       help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='multi_gpu_motor_detection', 
                       help='Experiment name')
    
    # Optimizer parameters
    parser.add_argument('--lr0', type=float, default=1e-4, 
                       help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.1, 
                       help='Final learning rate factor')
    parser.add_argument('--dropout', type=float, default=0.25, 
                       help='Dropout rate')
    
    # Data augmentation
    parser.add_argument('--mosaic', type=float, default=1.0, 
                       help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.5, 
                       help='MixUp augmentation probability')
    parser.add_argument('--degrees', type=float, default=45, 
                       help='Rotation degrees for augmentation')
    parser.add_argument('--scale', type=float, default=0.25, 
                       help='Scale augmentation factor')
    
    return parser.parse_args()

def setup_seeds(seed=2025):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_dataset(zip_path, extract_path):
    """Extract dataset from zip file"""
    if os.path.exists(zip_path):
        print(f"Extracting dataset from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset extraction completed!")
        return True
    else:
        print(f"Dataset zip file not found at {zip_path}")
        return False

def verify_and_convert_stack_images(images_dir):
    """
    Check if 2.5D stack images are 3-channel and convert grayscale to 3-channel if needed
    """
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
        
    converted_count = 0
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    
    for img_file in tqdm(image_files, desc="Converting images"):
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path)
        
        # If already 3-channel (2.5D stack)
        if len(np.array(img).shape) == 3 and np.array(img).shape[2] == 3:
            continue
        
        # If grayscale, convert to 3-channel
        elif len(np.array(img).shape) == 2 or (len(np.array(img).shape) == 3 and np.array(img).shape[2] == 1):
            img_array = np.array(img)
            if len(img_array.shape) == 2:
                # Replicate grayscale to 3 channels
                img_3ch = np.stack([img_array, img_array, img_array], axis=-1)
            else:
                # Replicate 1 channel to 3 channels
                img_3ch = np.repeat(img_array, 3, axis=-1)
            
            img_3ch_pil = Image.fromarray(img_3ch.astype(np.uint8))
            img_3ch_pil.save(img_path)
            converted_count += 1
            
    print(f"Converted {converted_count} images to 3-channel format")

def setup_dataset(data_path):
    """Setup dataset structure and create YAML configuration"""
    
    # Check for BYU subdirectory structure (from message.txt output)
    yolo_images_byu = os.path.join(data_path, "images", "byu")
    yolo_labels_byu = os.path.join(data_path, "labels", "byu")
    
    # Fallback to direct images/labels structure
    yolo_images_dir = os.path.join(data_path, "images")
    yolo_labels_dir = os.path.join(data_path, "labels")
    
    # Determine which structure exists
    if os.path.exists(yolo_images_byu):
        images_dir = yolo_images_byu
        labels_dir = yolo_labels_byu
        image_path_prefix = "images/byu"
        print("Using BYU subdirectory structure")
    elif os.path.exists(yolo_images_dir):
        images_dir = yolo_images_dir
        labels_dir = yolo_labels_dir
        image_path_prefix = "images"
        print("Using direct images/labels structure")
    else:
        raise FileNotFoundError(f"No images directory found in {data_path}")
    
    # Verify and convert images
    verify_and_convert_stack_images(images_dir)
    
    # Get all image files
    all_image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"Total images found: {len(all_image_files)}")
    
    # Create relative paths for train/val split
    all_image_files_relative = [str(Path(image_path_prefix) / f) for f in all_image_files]
    
    # Train/Val split
    np.random.shuffle(all_image_files_relative)
    split_idx = int(0.8 * len(all_image_files_relative))
    train_files = all_image_files_relative[:split_idx]
    val_files = all_image_files_relative[split_idx:]
    
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    
    # Save train/val file lists
    with open(os.path.join(data_path, 'train.txt'), 'w') as f:
        for file_path in train_files:
            f.write(f"./{file_path.replace(os.sep, '/')}\n")
    
    with open(os.path.join(data_path, 'val.txt'), 'w') as f:
        for file_path in val_files:
            f.write(f"./{file_path.replace(os.sep, '/')}\n")
    
    # Create YAML configuration
    yaml_content = {
        'path': str(Path(data_path).resolve()),
        'train': 'train.txt',
        'val': 'val.txt',
        'nc': 1,
        'names': {0: 'motor'}
    }
    
    yaml_path = os.path.join(data_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"Dataset YAML created at: {yaml_path}")
    return yaml_path

def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for train and validation, marking the best model
    """
    results_csv = os.path.join(run_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return None
    
    # Read results CSV
    results_df = pd.read_csv(results_csv)
    
    # Check if DFL loss columns exist
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        return None
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    # Find the epoch with the best validation loss
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation losses
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss', linewidth=2)
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss', linewidth=2)
    
    # Mark the best model with a vertical line
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', linewidth=2,
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    
    # Add labels and legend
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('DFL Loss', fontsize=12)
    plt.title('Multi-GPU Training: DFL Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss curve saved to {plot_path}")
    return best_epoch, best_val_loss

def main():
    """Main training function"""
    args = setup_args()
    
    # Setup random seeds
    setup_seeds()
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU(s) for training.")
    
    num_gpus = len(args.device.split(','))
    print(f"Using {num_gpus} GPU(s): {args.device}")
    
    # Calculate per-GPU batch size
    per_gpu_batch_size = args.batch_size // num_gpus
    print(f"Total batch size: {args.batch_size}")
    print(f"Per-GPU batch size: {per_gpu_batch_size}")
    
    # Check if dataset needs to be extracted
    if not os.path.exists(args.data_path):
        zip_path = f"{args.data_path}.zip"
        if extract_dataset(zip_path, "./"):
            print("Dataset extracted successfully")
        else:
            raise FileNotFoundError(f"Dataset not found at {args.data_path} or {zip_path}")
    
    # Setup dataset
    yaml_path = setup_dataset(args.data_path)
    
    # Load YOLO11m model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Configure training arguments
    train_args = {
        "data": yaml_path,
        "epochs": args.epochs,
        "batch": per_gpu_batch_size,  # Per-GPU batch size for DDP
        "imgsz": args.imgsz,
        "device": args.device,  # Multi-GPU device specification
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        
        # Optimizer settings
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "warmup_epochs": 0,
        "dropout": args.dropout,
        
        # Training management
        "patience": 100,
        "save_period": 5,
        "val": True,
        "verbose": True,
        "cache": False,  # Disable caching for large datasets
        
        # Data augmentation
        "mosaic": args.mosaic,
        "close_mosaic": 10,
        "mixup": args.mixup,
        "flipud": 0.5,
        "scale": args.scale,
        "degrees": args.degrees,
        
        # DDP specific settings
        "rect": False,  # Disable rectangular training for DDP
        "single_cls": False,
    }
    
    print("\n" + "="*50)
    print("STARTING MULTI-GPU TRAINING")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Dataset: {yaml_path}")
    print(f"GPUs: {args.device}")
    print(f"Total Batch Size: {args.batch_size}")
    print(f"Per-GPU Batch Size: {per_gpu_batch_size}")
    print(f"Workers per GPU: {args.workers}")
    print(f"Image Size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print("="*50 + "\n")
    
    # Start training
    results = model.train(**train_args)
    
    # Plot loss curves after training
    run_dir = os.path.join(args.project, args.name)
    best_epoch_info = plot_dfl_loss_curve(run_dir)
    
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
    # Save final model
    model_save_path = os.path.join(run_dir, "final_model.pt")
    model.save(model_save_path)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED!")
    print("="*50)
    print(f"Results saved to: {run_dir}")
    print(f"Best weights: {os.path.join(run_dir, 'weights', 'best.pt')}")
    print(f"Final model: {model_save_path}")
    print("="*50)

if __name__ == "__main__":
    main()
