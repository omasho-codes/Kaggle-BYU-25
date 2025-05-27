import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm
import cv2

# Set random seed for reproducibility
np.random.seed(42)

# Define Kaggle paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Define YOLO dataset structure for 2.5D
yolo_dataset_dir = "/kaggle/working/yolo_25d_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Define constants for 2.5D
TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
NUM_SLICES = 2 * TRUST + 1  # Total slices for multi-slice input (9 slices)
BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
TARGET_SIZE = 960  # Target image size for YOLO

# Image processing functions
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)
    
    # Normalize to [0, 255] range
    if p98 > p2:
        normalized = 255 * (clipped_data - p2) / (p98 - p2)
    else:
        normalized = slice_data
    
    return np.uint8(normalized)

def load_and_stack_slices(tomo_id, z_center, z_max, target_size=TARGET_SIZE):
    """
    Load and stack multiple slices around z_center for 2.5D input
    Returns stacked volume as (height, width, num_slices) numpy array
    """
    z_min = max(0, z_center - TRUST)
    z_max_slice = min(z_max - 1, z_center + TRUST)
    
    slices = []
    
    for z in range(z_min, z_max_slice + 1):
        slice_filename = f"slice_{z:04d}.jpg"
        slice_path = os.path.join(train_dir, tomo_id, slice_filename)
        
        if os.path.exists(slice_path):
            # Load and normalize slice
            img = Image.open(slice_path)
            img_array = np.array(img)
            normalized_slice = normalize_slice(img_array)
            slices.append(normalized_slice)
        else:
            # If slice doesn't exist, duplicate the last valid slice or create zeros
            if slices:
                slices.append(slices[-1].copy())
            else:
                # Create a zero slice with typical dimensions
                slices.append(np.zeros((928, 928), dtype=np.uint8))
    
    if not slices:
        return None, 0
    
    # Ensure we have exactly NUM_SLICES
    while len(slices) < NUM_SLICES:
        if slices:
            slices.append(slices[-1].copy())  # Duplicate last slice
        else:
            slices.append(np.zeros((928, 928), dtype=np.uint8))
    
    if len(slices) > NUM_SLICES:
        # Take center slices
        start_idx = (len(slices) - NUM_SLICES) // 2
        slices = slices[start_idx:start_idx + NUM_SLICES]
    
    # Stack slices along the last dimension
    volume = np.stack(slices, axis=-1)  # Shape: (height, width, num_slices)
    
    # Resize to target size while maintaining aspect ratio
    if target_size != volume.shape[0]:
        resized_slices = []
        for i in range(volume.shape[2]):
            resized_slice = cv2.resize(volume[:, :, i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            resized_slices.append(resized_slice)
        volume = np.stack(resized_slices, axis=-1)
    
    return volume, len([s for s in slices if s is not None])

def save_multislice_image(volume, filepath):
    """
    Save multi-slice volume as a multi-channel image
    For YOLO compatibility, we'll save as individual channels in a single file
    """
    if volume is None:
        return False
    
    # Convert to PIL Image format
    # We'll save as RGB by selecting 3 slices or grayscale
    if volume.shape[2] >= 3:
        # Use first, middle, and last slice as RGB channels
        middle_idx = volume.shape[2] // 2
        rgb_volume = np.stack([
            volume[:, :, 0],           # R: first slice
            volume[:, :, middle_idx],  # G: middle slice  
            volume[:, :, -1]           # B: last slice
        ], axis=-1)
        img = Image.fromarray(rgb_volume, mode='RGB')
    else:
        # Use first slice as grayscale
        img = Image.fromarray(volume[:, :, 0], mode='L')
    
    img.save(filepath)
    return True

def save_multislice_npy(volume, filepath):
    """
    Save multi-slice volume as numpy array (.npy file)
    This preserves all slice information for true 2.5D training
    """
    if volume is None:
        return False
    
    # Save as numpy array with shape (height, width, num_slices)
    np.save(filepath, volume.astype(np.float32) / 255.0)  # Normalize to [0, 1]
    return True

def prepare_yolo_25d_dataset(trust=TRUST, train_split=TRAIN_SPLIT, save_format='npy'):
    """
    Extract multi-slice volumes containing motors and save to YOLO structure
    
    Args:
        trust: Number of slices above and below center to include
        train_split: Ratio of data to use for training vs validation  
        save_format: 'npy' for numpy arrays (recommended for 2.5D) or 'jpg' for images
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    
    print(f"Found {len(unique_tomos)} unique tomograms with motors")
    print(f"Each sample will contain {NUM_SLICES} slices (trust={trust})")
    print(f"Target image size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Save format: {save_format}")
    
    # Perform the train-val split at the tomogram level
    np.random.shuffle(unique_tomos)
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    
    # Function to process a set of tomograms
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process {len(motor_counts)} motor volumes for {set_name}")
        
        # Process each motor
        processed_volumes = 0
        failed_volumes = 0
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            
            # Load multi-slice volume
            volume, valid_slices = load_and_stack_slices(tomo_id, z_center, z_max, TARGET_SIZE)
            
            if volume is None:
                print(f"Warning: Failed to load volume for {tomo_id} at z={z_center}")
                failed_volumes += 1
                continue
            
            # Create destination filename
            if save_format == 'npy':
                dest_filename = f"{tomo_id}_z{z_center:04d}_y{y_center:04d}_x{x_center:04d}.npy"
                dest_path = os.path.join(images_dir, dest_filename)
                success = save_multislice_npy(volume, dest_path)
            else:  # jpg format
                dest_filename = f"{tomo_id}_z{z_center:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                success = save_multislice_image(volume, dest_path)
            
            if not success:
                failed_volumes += 1
                continue
            
            # Calculate normalized coordinates for the resized image
            scale_factor = TARGET_SIZE / volume.shape[0]  # Assuming square images
            x_center_scaled = x_center * scale_factor
            y_center_scaled = y_center * scale_factor
            
            # Create YOLO format label
            x_center_norm = x_center_scaled / TARGET_SIZE
            y_center_norm = y_center_scaled / TARGET_SIZE
            box_width_norm = (BOX_SIZE * scale_factor) / TARGET_SIZE
            box_height_norm = (BOX_SIZE * scale_factor) / TARGET_SIZE
            
            # Ensure coordinates are within bounds
            x_center_norm = np.clip(x_center_norm, 0, 1)
            y_center_norm = np.clip(y_center_norm, 0, 1)
            box_width_norm = np.clip(box_width_norm, 0, 1)
            box_height_norm = np.clip(box_height_norm, 0, 1)
            
            # Write label file
            label_filename = dest_filename.replace('.npy', '.txt').replace('.jpg', '.txt')
            label_path = os.path.join(labels_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
            
            processed_volumes += 1
        
        if failed_volumes > 0:
            print(f"Warning: {failed_volumes} volumes failed to process in {set_name} set")
        
        return processed_volumes, len(motor_counts)
    
    # Process training tomograms
    train_volumes, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    
    # Process validation tomograms
    val_volumes, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Create YAML configuration file for YOLO
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'},
        'nc': 1,  # Number of classes
        # Add 2.5D specific metadata
        'multislice': True,
        'num_slices': NUM_SLICES,
        'trust': TRUST,
        'target_size': TARGET_SIZE,
        'save_format': save_format
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    # Create a README file with important information
    readme_content = f"""# 2.5D Flagellar Motor Dataset

## Dataset Information
- Format: {save_format}
- Input channels: {NUM_SLICES} slices per volume
- Trust parameter: {TRUST} (slices above/below center)
- Target image size: {TARGET_SIZE}x{TARGET_SIZE}
- Box size: {BOX_SIZE}

## Training Data
- Tomograms: {len(train_tomos)}
- Motors: {train_motors}
- Volumes: {train_volumes}

## Validation Data  
- Tomograms: {len(val_tomos)}
- Motors: {val_motors}
- Volumes: {val_volumes}

## Usage for YOLO Training
1. Modify YOLO model input channels from 3 to {NUM_SLICES}
2. Use custom dataloader that loads .npy files if save_format='npy'
3. Ensure proper normalization during training

## File Structure
- images/train/: Training volumes ({save_format} files)
- images/val/: Validation volumes ({save_format} files)  
- labels/train/: Training labels (YOLO format .txt files)
- labels/val/: Validation labels (YOLO format .txt files)
- dataset.yaml: YOLO configuration file
"""
    
    with open(os.path.join(yolo_dataset_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_volumes} volumes")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_volumes} volumes")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_volumes + val_volumes} volumes")
    print(f"- Multi-slice format: {NUM_SLICES} slices per volume")
    print(f"- Save format: {save_format}")
    
    # Return summary info
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_volumes": train_volumes,
        "val_volumes": val_volumes,
        "num_slices": NUM_SLICES,
        "target_size": TARGET_SIZE,
        "save_format": save_format
    }

# Run the preprocessing for 2.5D
print("=" * 60)
print("2.5D Multi-slice YOLO Dataset Preparation")
print("=" * 60)

# Use 'npy' format for true 2.5D (preserves all slice information)
# Use 'jpg' format for compatibility but loses some slice information
summary = prepare_yolo_25d_dataset(TRUST, save_format='npy')

print(f"\n" + "=" * 60)
print("Preprocessing Complete!")
print("=" * 60)
print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_volumes']} volumes")
print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_volumes']} volumes")
print(f"- Multi-slice input: {summary['num_slices']} slices per volume")
print(f"- Target size: {summary['target_size']}x{summary['target_size']}")
print(f"- Save format: {summary['save_format']}")
print(f"- Dataset directory: {summary['dataset_dir']}")
print(f"- YAML configuration: {summary['yaml_path']}")
print(f"\nReady for 2.5D YOLO training!")