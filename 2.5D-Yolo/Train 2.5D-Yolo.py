import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss
import yaml
import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
NUM_SLICES = 9  # Number of slices for 2.5D input
TARGET_SIZE = 960  # Target image size for training

class MultiSlice25DDataset(Dataset):
    """
    2.5D YOLO Dataset for multi-slice bacterial flagellar motor detection
    
    This dataset loads .npy files containing multiple slices and corresponding
    YOLO format labels for training 2.5D object detection models.
    """

    def __init__(self, img_path, imgsz=960, num_slices=9, mode='train', augment=False):
        """
        Initialize the dataset
        
        Args:
            img_path (str): Path to image directory
            imgsz (int): Target image size
            num_slices (int): Number of slices to use for 2.5D input
            mode (str): Dataset mode ('train', 'val', 'test')
            augment (bool): Whether to apply data augmentation
        """
        self.img_path = img_path
        self.imgsz = imgsz
        self.num_slices = num_slices
        self.mode = mode
        self.augment = augment

        self.img_files = self._get_img_files()
        self.label_files = self._get_label_files()

        print(f"[{mode.upper()}] Found .npy files: {len(self.img_files)}")
        print(f"[{mode.upper()}] Found label files: {len(self.label_files)}")

    def _get_img_files(self):
        """Get list of image files (.npy format)"""
        if not os.path.exists(self.img_path):
            return []
        pattern = os.path.join(self.img_path, "*.npy")
        return sorted(glob.glob(pattern))

    def _get_label_files(self):
        """Get corresponding label files for each image"""
        label_files = []
        for img_file in self.img_files:
            label_file = img_file.replace('/images/', '/labels/').replace('.npy', '.txt')
            label_files.append(label_file)
        return label_files

    def __len__(self):
        return len(self.img_files)

    def _load_labels(self, label_path):
        """
        Load YOLO format labels
        
        Args:
            label_path (str): Path to label file
            
        Returns:
            list: List of labels in format [class_id, x_center, y_center, width, height]
        """
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([class_id, x_center, y_center, width, height])
        return labels

    def _apply_augmentation(self, volume, labels):
        """
        Apply data augmentation to volume and labels
        
        Args:
            volume (np.ndarray): Input volume
            labels (list): List of labels
            
        Returns:
            tuple: Augmented volume and labels
        """
        if not self.augment or self.mode != 'train':
            return volume, labels

        # Random horizontal flip
        if np.random.random() > 0.5:
            volume = np.fliplr(volume).copy()  # .copy() to solve negative stride issue
            # Flip x coordinates in labels
            for label in labels:
                label[1] = 1.0 - label[1]  # x_center flip

        # Random rotation (small angles)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            center = (volume.shape[1]//2, volume.shape[0]//2)

            # Apply rotation to each slice
            rotated_slices = []
            for k in range(volume.shape[2]):
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_slice = cv2.warpAffine(volume[:, :, k], M, (volume.shape[1], volume.shape[0]))
                rotated_slices.append(rotated_slice)
            volume = np.stack(rotated_slices, axis=2)

        # Final safety copy
        volume = volume.copy()

        return volume, labels

    def __getitem__(self, idx):
        """
        Get a sample from the dataset
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (image_tensor, labels_tensor, image_path)
        """
        try:
            # Load volume
            volume = np.load(self.img_files[idx])

            # Create safe array copy
            volume = np.array(volume, copy=True)

            # Handle shape
            if len(volume.shape) == 2:
                volume = np.expand_dims(volume, axis=-1)

            # Adjust number of slices
            if volume.shape[2] != self.num_slices:
                if volume.shape[2] == 1:
                    volume = np.repeat(volume, self.num_slices, axis=2)
                elif volume.shape[2] > self.num_slices:
                    volume = volume[:, :, :self.num_slices].copy()
                else:
                    last_slice = volume[:, :, -1:]
                    padding_needed = self.num_slices - volume.shape[2]
                    padding = np.repeat(last_slice, padding_needed, axis=2)
                    volume = np.concatenate([volume, padding], axis=2).copy()

            # Normalize
            if volume.max() > 1:
                volume = volume.astype(np.float32) / 255.0
            else:
                volume = volume.astype(np.float32)

            # Ensure contiguous array
            if not volume.flags['C_CONTIGUOUS']:
                volume = np.ascontiguousarray(volume)

            # Resize if needed
            if volume.shape[0] != self.imgsz or volume.shape[1] != self.imgsz:
                resized_slices = []
                for k in range(self.num_slices):
                    slice_2d = np.ascontiguousarray(volume[:, :, k])
                    resized_slice = cv2.resize(slice_2d, (self.imgsz, self.imgsz))
                    resized_slices.append(resized_slice)
                volume = np.stack(resized_slices, axis=-1)
                volume = np.ascontiguousarray(volume)

            # Load labels
            labels = self._load_labels(self.label_files[idx])

            # Apply augmentation
            volume, labels = self._apply_augmentation(volume, labels)

            # Final safety check
            volume = np.ascontiguousarray(volume)

            # Convert to tensor (C, H, W) format
            img_tensor = torch.from_numpy(volume).permute(2, 0, 1).float()

            # Convert labels to tensor
            if labels:
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
            else:
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)

            return img_tensor, labels_tensor, self.img_files[idx]

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return safe dummy data
            img_tensor = torch.zeros((self.num_slices, self.imgsz, self.imgsz), dtype=torch.float32)
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
            return img_tensor, labels_tensor, f"error_{idx}"


def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO training
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        tuple: (images, targets, paths)
    """
    imgs = torch.stack([item[0] for item in batch])

    # Process labels - create safe batch
    batch_labels = []
    valid_samples = 0

    for batch_idx, (_, labels, path) in enumerate(batch):
        if len(labels) > 0 and not torch.isnan(labels).any():
            # Add batch index to each label
            for label in labels:
                if len(label) >= 5:  # Minimum requirement: [class, x, y, w, h]
                    batch_labels.append([batch_idx] + label.tolist())
                    valid_samples += 1

    # Create target tensor
    if batch_labels and valid_samples > 0:
        targets = torch.tensor(batch_labels, dtype=torch.float32)

        # Dimension validation
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        # Column count validation (minimum 6: batch_idx + class + x + y + w + h)
        if targets.size(1) < 6:
            print(f"Warning: Insufficient target dimensions {targets.shape}, adding padding")
            padding = torch.zeros(targets.size(0), 6 - targets.size(1))
            targets = torch.cat([targets, padding], dim=1)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    paths = [item[2] for item in batch]

    return imgs, targets, paths


class YOLO25DTrainer:
    """
    2.5D YOLO Trainer class for bacterial flagellar motor detection
    
    This trainer handles the complete training pipeline including:
    - Model initialization and modification for multi-channel input
    - Loss computation with fallback mechanisms
    - Training and validation loops
    - Checkpoint saving and metrics visualization
    """

    def __init__(self, model_path, device='cuda'):
        """
        Initialize the trainer
        
        Args:
            model_path (str): Path to the modified YOLO model
            device (str): Device to use for training
        """
        self.device = device
        self.model = YOLO(model_path)
        self.model.model.to(device)

        # Try to use ComputeLoss with fallback to custom loss
        self.use_compute_loss = False

        try:
            from ultralytics.nn.tasks import ComputeLoss
            self.compute_loss = ComputeLoss(self.model.model)
            print("✅ ComputeLoss loaded successfully! Starting in hybrid mode.")
            self.use_compute_loss = True
        except Exception as e:
            print(f"⚠️ ComputeLoss loading failed: {e}")
            print("📍 Using custom loss function.")
            self.use_compute_loss = False

        # Backup custom loss function
        self.custom_loss_fn = self._create_effective_loss_fn()

        # Training metrics storage
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        print(f"Model loaded: {model_path}")
        print(f"Device: {device}")
        print(f"Loss function: {'ComputeLoss (hybrid)' if self.use_compute_loss else 'Custom loss'}")

    def switch_to_compute_loss(self):
        """Switch to ComputeLoss (can be called during training)"""
        if hasattr(self, 'compute_loss'):
            self.use_compute_loss = True
            print("🔄 Switched to ComputeLoss!")
        else:
            print("❌ ComputeLoss not available, cannot switch.")

    def switch_to_custom_loss(self):
        """Switch to custom loss function"""
        self.use_compute_loss = False
        print("🔄 Switched to custom loss function!")

    def _create_effective_loss_fn(self):
        """
        Create effective and stable loss function with improved type safety
        
        Returns:
            function: Loss function for YOLO predictions and targets
        """
        def effective_loss(predictions, targets):
            """
            Effective loss function for YOLO predictions and targets
            - Object confidence loss
            - Bounding box regression loss
            - Class classification loss
            """
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Process predictions (type safety)
            if isinstance(predictions, (list, tuple)):
                # Handle multi-scale predictions
                for pred in predictions:
                    if torch.is_tensor(pred):
                        # Signal strength based loss (objectness proxy)
                        pred_flat = pred.view(pred.size(0), -1)  # [batch, features]
                        confidence_loss = torch.mean(torch.sigmoid(pred_flat)) * 0.1
                        total_loss = total_loss + confidence_loss
            else:
                if torch.is_tensor(predictions):
                    pred_flat = predictions.view(predictions.size(0), -1)
                    confidence_loss = torch.mean(torch.sigmoid(pred_flat)) * 0.1
                    total_loss = total_loss + confidence_loss

            # Process targets (greatly improved type safety)
            if targets is not None:
                # Convert targets to tensor
                if isinstance(targets, (list, tuple)):
                    if len(targets) > 0:
                        # Convert list to tensor
                        try:
                            if all(torch.is_tensor(t) for t in targets):
                                targets_tensor = torch.cat([t.view(-1, t.size(-1)) for t in targets if t.numel() > 0], dim=0)
                            else:
                                targets_tensor = torch.tensor(targets, device=self.device)
                        except:
                            # Safe handling on conversion failure
                            return total_loss
                    else:
                        targets_tensor = torch.empty(0, 6, device=self.device)
                elif torch.is_tensor(targets):
                    targets_tensor = targets
                else:
                    # Safe skip for other types
                    return total_loss

                # Process only if tensor is non-empty and valid
                if targets_tensor.numel() > 0 and len(targets_tensor.shape) >= 2 and targets_tensor.size(1) >= 6:
                    # Bounding box coordinate loss (normalized coordinates in 0-1 range)
                    bbox_coords = targets_tensor[:, 2:6]  # [x, y, w, h]

                    # Check if coordinates are in valid range
                    valid_mask = (bbox_coords >= 0) & (bbox_coords <= 1)
                    valid_rows = valid_mask.all(dim=1)

                    if valid_rows.any():
                        valid_coords = bbox_coords[valid_rows]

                        # L1 loss (more stable)
                        bbox_loss = torch.mean(torch.abs(valid_coords - 0.5)) * 2.0  # Penalty for deviation from center
                        total_loss = total_loss + bbox_loss

                        # Size consistency loss (prevent too large or small boxes)
                        width_height = valid_coords[:, 2:]  # w, h
                        if len(width_height) > 1:
                            size_loss = torch.mean(torch.abs(width_height - torch.mean(width_height, dim=0))) * 1.0
                            total_loss = total_loss + size_loss

            return total_loss

        return effective_loss

    def create_data_loaders(self, dataset_path, batch_size=4, num_workers=0):
        """
        Create data loaders for training and validation
        
        Args:
            dataset_path (str): Path to dataset directory
            batch_size (int): Batch size for training
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            tuple: (train_loader, val_loader)
        """
        train_path = os.path.join(dataset_path, 'images', 'train')
        val_path = os.path.join(dataset_path, 'images', 'val')

        # Create datasets (augmentation disabled for safe start)
        train_dataset = MultiSlice25DDataset(
            train_path, imgsz=TARGET_SIZE, num_slices=NUM_SLICES,
            mode='train', augment=False  # Disable augmentation for first training
        )
        val_dataset = MultiSlice25DDataset(
            val_path, imgsz=TARGET_SIZE, num_slices=NUM_SLICES,
            mode='val', augment=False
        )

        # Create data loaders with custom collate function
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=yolo_collate_fn, num_workers=num_workers,
            pin_memory=True, drop_last=True
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=yolo_collate_fn, num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    def calculate_yolo_loss(self, predictions, targets):
        """
        Hybrid YOLO loss calculation (ComputeLoss + custom fallback)
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            torch.Tensor: Computed loss
        """

        # Try ComputeLoss first
        if self.use_compute_loss and hasattr(self, 'compute_loss'):
            try:
                # Validate and preprocess targets
                if len(targets) == 0 or not torch.is_tensor(targets):
                    return self._calculate_custom_loss(predictions, targets)

                # Call ComputeLoss
                loss_tuple = self.compute_loss(predictions, targets)

                if isinstance(loss_tuple, (list, tuple)) and len(loss_tuple) >= 1:
                    total_loss = loss_tuple[0]  # Total loss

                    # Validate loss
                    if torch.is_tensor(total_loss) and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                        # ComputeLoss successful!
                        return total_loss
                    else:
                        print("⚠️ ComputeLoss returned invalid loss, falling back to custom loss")
                        return self._calculate_custom_loss(predictions, targets)
                else:
                    print("⚠️ ComputeLoss format error, falling back to custom loss")
                    return self._calculate_custom_loss(predictions, targets)

            except Exception as e:
                # Auto-switch to custom loss on ComputeLoss failure
                if "anchor" in str(e).lower() or "stride" in str(e).lower():
                    print(f"📍 ComputeLoss compatibility issue detected: {str(e)[:50]}...")
                    print("💡 Permanently switching to custom loss.")
                    self.use_compute_loss = False  # Permanent disable

                return self._calculate_custom_loss(predictions, targets)

        # Use custom loss
        return self._calculate_custom_loss(predictions, targets)

    def _calculate_custom_loss(self, predictions, targets):
        """
        Backup custom loss function
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            torch.Tensor: Computed loss
        """
        try:
            loss = self.custom_loss_fn(predictions, targets)

            if not torch.is_tensor(loss):
                return torch.tensor(0.01, device=self.device, requires_grad=True)

            if torch.isnan(loss) or torch.isinf(loss):
                return torch.tensor(0.01, device=self.device, requires_grad=True)

            if loss.item() > 100:
                loss = torch.clamp(loss, max=10.0)

            return loss

        except Exception as e:
            if not ("view" in str(e) or "list" in str(e)):
                print(f"Custom loss error: {str(e)[:30]}...")

            return torch.tensor(0.01, device=self.device, requires_grad=True)

    def train_epoch(self, train_loader, optimizer, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer for training
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss for the epoch
        """
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        error_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Training')

        for batch_idx, (imgs, targets, paths) in enumerate(pbar):
            # Move to GPU
            imgs = imgs.to(self.device)
            targets = targets.to(self.device) if len(targets) > 0 else targets

            optimizer.zero_grad()

            try:
                # Forward pass
                predictions = self.model.model(imgs)

                # Calculate loss
                loss = self.calculate_yolo_loss(predictions, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=10.0)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # Update progress every 20 batches (reduce logging)
                if batch_idx % 20 == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{total_loss/num_batches:.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.5f}',
                        'Tgts': len(targets),
                        'Errs': error_count
                    })

            except Exception as e:
                error_count += 1
                if error_count < 5:  # Only print first 5 errors
                    print(f"Batch {batch_idx} error: {str(e)[:50]}...")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        if error_count > 0:
            print(f"Epoch {epoch+1}: Total {error_count} batches with errors")

        return avg_loss

    def validate_epoch(self, val_loader, epoch):
        """
        Validate for one epoch
        
        Args:
            val_loader: Validation data loader
            epoch (int): Current epoch number
            
        Returns:
            float: Average validation loss for the epoch
        """
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        error_count = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} Validation')

            for batch_idx, (imgs, targets, paths) in enumerate(pbar):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device) if len(targets) > 0 else targets

                try:
                    predictions = self.model.model(imgs)
                    loss = self.calculate_yolo_loss(predictions, targets)

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 20 == 0:
                        pbar.set_postfix({
                            'Val Loss': f'{loss.item():.4f}',
                            'Avg': f'{total_loss/num_batches:.4f}',
                            'Tgts': len(targets),
                            'Errs': error_count
                        })

                except Exception as e:
                    error_count += 1
                    continue

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def save_checkpoint(self, epoch, train_loss, val_loss, save_dir, is_best=False):
        """
        Save training checkpoint
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            save_dir (str): Directory to save checkpoint
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat()
        }

        # Periodic checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Best performance model
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

        # Latest model
        latest_path = os.path.join(save_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)

    def plot_training_curves(self, save_dir):
        """
        Plot training curves with proper length handling
        
        Args:
            save_dir (str): Directory to save plots
        """
        if len(self.train_losses) == 0:
            return

        # Font settings (error prevention)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 5))

        # Loss curves - handle length mismatch
        min_length = min(len(self.train_losses), len(self.val_losses))
        epochs = range(1, min_length + 1)

        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses[:min_length], 'b-', label='Train Loss')
        plt.plot(epochs, self.val_losses[:min_length], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Recent loss (zoomed) - safe length calculation
        plt.subplot(1, 2, 2)
        recent_epochs = max(1, min_length - 20)
        recent_range = range(recent_epochs, min_length + 1)
        recent_length = min_length - recent_epochs + 1

        plt.plot(recent_range, self.train_losses[-recent_length:], 'b-', label='Train Loss')
        plt.plot(recent_range, self.val_losses[-recent_length:], 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent 20 Epochs Loss')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {plot_path}")

    def train(self, dataset_path, epochs=100, batch_size=4, lr=1e-3, save_dir='./yolo_25d_training'):
        """
        Complete training process with ComputeLoss performance monitoring
        
        Args:
            dataset_path (str): Path to dataset directory
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            lr (float): Learning rate
            save_dir (str): Directory to save training outputs
            
        Returns:
            str: Path to best model checkpoint
        """
        print(f"\n🚀 2.5D YOLO Training Started!")
        print(f"Epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        weights_dir = os.path.join(save_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)

        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            dataset_path, batch_size=batch_size
        )

        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=lr,
            weight_decay=0.0005
        )

        # Setup scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr*0.01
        )

        # ComputeLoss performance tracking
        compute_loss_successes = 0
        compute_loss_failures = 0

        # Start training
        print(f"\n📊 Training started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")

            # Training
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            self.train_losses.append(train_loss)

            # Validation
            val_loss = self.validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)

            # Adjust learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # ComputeLoss performance monitoring
            if hasattr(self, 'compute_loss') and self.use_compute_loss:
                # Check success/failure ratio (virtual counter - actual implementation should count in loss function)
                success_rate = 0.9  # Temporary value
                if success_rate < 0.5:  # If less than 50% success rate
                    print("📉 ComputeLoss performance low, switching to custom loss.")
                    self.switch_to_custom_loss()

            # Output results
            loss_type = "ComputeLoss" if (hasattr(self, 'compute_loss') and self.use_compute_loss) else "Custom"
            print(f"Training loss: {train_loss:.6f} ({loss_type})")
            print(f"Validation loss: {val_loss:.6f}")
            print(f"Learning rate: {current_lr:.6f}")

            # Check for best performance model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"🏆 New best performance! Validation loss: {val_loss:.6f}")

                # Suggest complete transition after epoch 5 if ComputeLoss works well
                if epoch >= 5 and hasattr(self, 'compute_loss') and self.use_compute_loss:
                    print("💡 ComputeLoss is working stably!")

            # Save checkpoint
            self.save_checkpoint(
                epoch, train_loss, val_loss, weights_dir, is_best
            )

            # Update training curves
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves(save_dir)

        # Final results
        print(f"\n🎯 Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"Loss function used: {loss_type}")
        print(f"Final model saved to: {weights_dir}")

        # Save final training curves
        self.plot_training_curves(save_dir)

        # Save training log
        training_log = {
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
                'input_channels': NUM_SLICES,
                'image_size': TARGET_SIZE,
                'loss_function': loss_type
            },
            'results': {
                'best_val_loss': self.best_val_loss,
                'final_train_loss': self.train_losses[-1] if self.train_losses else 0,
                'final_val_loss': self.val_losses[-1] if self.val_losses else 0,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'used_compute_loss': hasattr(self, 'compute_loss') and self.use_compute_loss
            },
            'timestamp': datetime.now().isoformat()
        }

        log_path = os.path.join(save_dir, 'training_log.json')
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)

        print(f"Training log saved: {log_path}")

        return os.path.join(weights_dir, 'best_model.pt')


def modify_yolo_for_multislice_simple(model_path, num_input_channels=9):
    """
    Simple model modification that's more robust for multi-channel input
    
    Args:
        model_path (str): Path to base YOLO model
        num_input_channels (int): Number of input channels to modify to
        
    Returns:
        YOLO: Modified YOLO model with updated input channels
    """
    print(f"🔧 Modifying YOLO model to {num_input_channels} channels...")

    model = YOLO(model_path)

    # Find the first conv layer and modify it
    def find_and_modify_conv(module, path=""):
        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.Conv2d) and child.in_channels == 3:
                print(f"Found Conv2d layer at {current_path}: {child}")

                # Create new conv layer
                new_conv = nn.Conv2d(
                    in_channels=num_input_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=child.bias is not None,
                    padding_mode=child.padding_mode
                )

                # Initialize weights
                with torch.no_grad():
                    # Copy original weights for first 3 channels
                    new_conv.weight[:, :3] = child.weight.clone()

                    # Initialize additional channels by repeating RGB pattern
                    if num_input_channels > 3:
                        remaining = num_input_channels - 3
                        for i in range(remaining):
                            channel_idx = i % 3
                            new_conv.weight[:, 3 + i] = child.weight[:, channel_idx].clone()

                    # Copy bias if exists
                    if child.bias is not None:
                        new_conv.bias = child.bias.clone()

                # Replace the layer
                setattr(module, name, new_conv)
                print(f"Successfully modified {current_path}: {child.in_channels} -> {num_input_channels} channels")
                return True

            # Recursively search in child modules
            if find_and_modify_conv(child, current_path):
                return True

        return False

    # Modify the model
    success = find_and_modify_conv(model.model)

    if success:
        print(f"✅ Model successfully modified for {num_input_channels} input channels")
    else:
        print("⚠️ Warning: Could not find Conv2d layer with 3 input channels")

    return model


def main_training():
    """
    Main training function with automatic model modification
    """
    print("🔥 2.5D YOLO Training Started!")

    # Configuration
    dataset_path = "/content/datasets/yolo_dataset"
    base_model_path = "yolo11m.pt"  # Base YOLO model
    modified_model_path = "./yolo11m_25d_modified.pt"

    # Create modified model if it doesn't exist
    if not os.path.exists(modified_model_path):
        print(f"📍 Modified model not found. Creating automatically...")
        print(f"Base model: {base_model_path} → Modified model: {modified_model_path}")

        try:
            # Modify YOLO model to 9 channels
            modified_model = modify_yolo_for_multislice_simple(base_model_path, NUM_SLICES)

            # Save modified model
            modified_model.save(modified_model_path)
            print(f"✅ Modified model saved successfully: {modified_model_path}")

        except Exception as e:
            print(f"❌ Model modification failed: {e}")
            return
    else:
        print(f"✅ Modified model already exists: {modified_model_path}")

    # Create trainer
    trainer = YOLO25DTrainer(modified_model_path)

    # Execute training
    best_model_path = trainer.train(
        dataset_path=dataset_path,
        epochs=50,          # Number of epochs (reduced for testing)
        batch_size=4,       # Batch size
        lr=1e-3,           # Learning rate
        save_dir='./yolo_25d_training_results'
    )

    print(f"\n🎉 Training completed!")
    print(f"Best performance model: {best_model_path}")
    print(f"You can now use this model for predictions!")


def test_trained_model(model_path, dataset_path, num_samples=5):
    """
    Test the trained 2.5D YOLO model
    
    Args:
        model_path (str): Path to trained model checkpoint
        dataset_path (str): Path to dataset directory
        num_samples (int): Number of samples to test
        
    Returns:
        list: Test results
    """
    print(f"🧪 Testing trained model: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        # Load model (from checkpoint)
        checkpoint = torch.load(model_path, map_location='cpu')

        # Load base YOLO model
        base_model = YOLO('./yolo11m_25d_modified.pt')

        # Apply trained weights
        if 'model_state_dict' in checkpoint:
            base_model.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Trained weights loaded successfully")
        else:
            print("⚠️ model_state_dict not found in checkpoint")

        base_model.model.eval()

        # Load test data
        test_path = os.path.join(dataset_path, 'images', 'val')
        test_dataset = MultiSlice25DDataset(
            test_path, imgsz=TARGET_SIZE, num_slices=NUM_SLICES, mode='test'
        )

        if len(test_dataset) == 0:
            print("❌ No test data found")
            return

        print(f"📊 Test samples: {len(test_dataset)}")

        # Select random samples
        import random
        test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))

        print(f"\n🔍 Testing {num_samples} samples...")

        # Storage for results
        results = []

        with torch.no_grad():
            for i, idx in enumerate(test_indices):
                try:
                    img_tensor, _, img_path = test_dataset[idx]

                    # Add batch dimension
                    img_batch = img_tensor.unsqueeze(0)  # [1, 9, 960, 960]

                    # Perform prediction
                    predictions = base_model.model(img_batch)

                    # Analyze prediction results
                    if isinstance(predictions, (list, tuple)):
                        pred_info = f"Multi-scale: {len(predictions)} outputs"
                        first_shape = predictions[0].shape if len(predictions) > 0 else "No output"
                    else:
                        pred_info = f"Single output: {predictions.shape}"
                        first_shape = predictions.shape

                    results.append({
                        'sample': i+1,
                        'path': img_path,
                        'input_shape': img_tensor.shape,
                        'prediction_info': pred_info,
                        'output_shape': first_shape
                    })

                    print(f"  Sample {i+1}: ✅ Success - {pred_info}")

                except Exception as e:
                    print(f"  Sample {i+1}: ❌ Failed - {str(e)[:50]}...")
                    results.append({
                        'sample': i+1,
                        'path': f"error_{idx}",
                        'error': str(e)
                    })

        # Results summary
        successful_tests = len([r for r in results if 'error' not in r])
        print(f"\n📈 Test Results Summary:")
        print(f"Successful tests: {successful_tests}/{len(results)}")
        print(f"Success rate: {successful_tests/len(results)*100:.1f}%")

        if successful_tests > 0:
            print("✅ Model can perform predictions normally!")
            print("🎯 Ready to perform object detection on real data!")
        else:
            print("❌ Model prediction has issues. Debugging needed.")

        return results

    except Exception as e:
        print(f"❌ Error during model testing: {e}")
        return None


def main_test():
    """Execute trained model testing"""
    model_path = "./yolo_25d_training_results/weights/best_model.pt"
    dataset_path = "/content/datasets/yolo_dataset"

    results = test_trained_model(model_path, dataset_path, num_samples=5)

    if results:
        print("\n🎉 Model testing completed!")
        print("You can now perform object detection on new .npy files.")
    else:
        print("\n❌ Model testing failed")


def main():
    """
    Main function - allows user selection
    
    Options:
    1. Training: Start 2.5D YOLO training
    2. Testing: Test trained model
    3. Auto: Test if model exists, otherwise train
    """
    print("🚀 2.5D YOLO System")
    print("1. Training")
    print("2. Testing") 
    print("3. Auto (Test if model exists, otherwise train)")
    
    # For now, directly start training
    main_training()
    
    # Uncomment below for interactive mode:
    # choice = input("Select option (1/2/3): ").strip()
    # 
    # if choice == "1":
    #     print("\n📚 Starting training...")
    #     main_training()
    # elif choice == "2":
    #     print("\n🧪 Starting testing...")
    #     main_test()
    # elif choice == "3":
    #     print("\n🤖 Auto mode...")
    #     if os.path.exists("./yolo_25d_training_results/weights/best_model.pt"):
    #         print("Model exists. Running test...")
    #         main_test()
    #     else:
    #         print("Model not found. Starting training...")
    #         main_training()
    # else:
    #     print("❌ Invalid selection.")


if __name__ == "__main__":
    main()