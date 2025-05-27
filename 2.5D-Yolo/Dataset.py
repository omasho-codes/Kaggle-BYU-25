import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import cv2 # OpenCV, often a dependency for Albumentations for border modes etc.

import albumentations as A
# from albumentations.pytorch import ToTensorV2 # Can be used at the end of color augs

# --- Configuration Constants ---
BB_WIDTH_PX = 36
BB_HEIGHT_PX = 36

class Tomo25DDataset(Dataset):
    def __init__(self,
                 positive_labels_df: pd.DataFrame,
                 image_base_dir: str,
                 negative_coords_list: list = None,
                 tomo_shape_z_col: str = 'tomo_shape_z', # Expected col names in DataFrames
                 tomo_shape_h_col: str = 'tomo_shape_h',
                 tomo_shape_w_col: str = 'tomo_shape_w',
                 trust_param_positive: int = 3,
                 slice_selection_negative: str = 'z_pm_1', # 'z_pm_1' or 'z_repeat_3'
                 spatial_transform_fn=None, # Albumentations Compose object or similar
                 color_transform_fn=None,   # Albumentations Compose object or similar
                 is_train: bool = True,
                 slicename_format: str = "{:04d}.jpg",
                 normalize_output_range: bool = True): # True for 0-1, False for 0-255 tensor

        self.image_base_dir = image_base_dir
        self.tomo_shape_z_col = tomo_shape_z_col
        self.tomo_shape_h_col = tomo_shape_h_col
        self.tomo_shape_w_col = tomo_shape_w_col
        self.trust_param_positive = trust_param_positive
        self.slice_selection_negative = slice_selection_negative
        
        self.is_train = is_train
        self.spatial_transform = spatial_transform_fn
        self.color_transform = color_transform_fn
        
        self.slicename_format = slicename_format
        self.normalize_output_range = normalize_output_range

        self.samples = self._prepare_samples(positive_labels_df, negative_coords_list)
        if not self.samples:
            raise ValueError("No samples were prepared. Check input data and parameters.")

    def _get_tomo_shape_from_row(self, sample_row_or_dict):
        return (int(sample_row_or_dict[self.tomo_shape_z_col]),
                int(sample_row_or_dict[self.tomo_shape_h_col]),
                int(sample_row_or_dict[self.tomo_shape_w_col]))

    def _prepare_samples(self, pos_df: pd.DataFrame, neg_list: list = None):
        final_samples_list = []
        
        if pos_df is None or pos_df.empty:
            print("Warning: Positive labels DataFrame is empty or None.")
        else:
            valid_pos_df = pos_df[pos_df['num_motors'] > 0].copy()
            if valid_pos_df.empty:
                print("Warning: No positive labels found after filtering for num_motors > 0.")
            else:
                tomogram_gt_counts = valid_pos_df['tomogram_id'].value_counts()
                
                # Group by (tomogram_id, z) to collect all GTs at each relevant z-plane
                # These z-planes are the 'primary_z_for_sampling'
                grouped_pos_by_z = valid_pos_df.groupby(['tomogram_id', 'z'])

                for (tomo_id, primary_z), group_df in grouped_pos_by_z:
                    first_row = group_df.iloc[0]
                    tomo_shape_z, tomo_shape_h, tomo_shape_w = self._get_tomo_shape_from_row(first_row)
                    
                    # is_multi_motor_context is based on total GTs in the tomogram, not just this z-plane
                    is_multi_context = tomogram_gt_counts.get(tomo_id, 0) > 1
                    
                    gt_objects_px_at_primary_z = []
                    for _, gt_row in group_df.iterrows():
                        gt_objects_px_at_primary_z.append({
                            'class_id': 0, # YOLO class ID for motors
                            'x_center_px': int(gt_row['x']), 
                            'y_center_px': int(gt_row['y']),
                            # width and height are fixed, not from df
                        })
                    
                    if not gt_objects_px_at_primary_z: continue # Should not happen due to groupby logic

                    final_samples_list.append({
                        'tomogram_id': tomo_id,
                        'primary_z_for_sampling': int(primary_z),
                        'tomo_shape_z': tomo_shape_z, 'tomo_shape_h': tomo_shape_h, 'tomo_shape_w': tomo_shape_w,
                        'is_multi_motor_context': is_multi_context,
                        'gt_objects_centers_px': gt_objects_px_at_primary_z, # List of {'class_id', 'x_center_px', 'y_center_px'}
                        'label_type': 'positive'
                    })

        # Process negative samples
        if neg_list:
            for neg_item in neg_list:
                try:
                    tomo_shape_z, tomo_shape_h, tomo_shape_w = self._get_tomo_shape_from_row(neg_item)
                    center_z_neg = int(neg_item['z'])
                    if not (0 <= center_z_neg < tomo_shape_z):
                        print(f"Warning: Negative Z-coord {center_z_neg} for tomo {neg_item['tomogram_id']} out of bounds. Skipping.")
                        continue
                    final_samples_list.append({
                        'tomogram_id': neg_item['tomogram_id'],
                        'primary_z_for_sampling': center_z_neg,
                        'tomo_shape_z': tomo_shape_z, 'tomo_shape_h': tomo_shape_h, 'tomo_shape_w': tomo_shape_w,
                        'is_multi_motor_context': False, # Not applicable
                        'gt_objects_centers_px': [], # Empty for negatives
                        'label_type': 'negative'
                    })
                except KeyError as e:
                    print(f"Warning: Missing key {e} in negative sample item: {neg_item}. Skipping.")
                    continue
        return final_samples_list

    def _get_positive_slice_indices(self, primary_z: int, max_z_idx: int, is_multi_context: bool, trust: int):
        clamped_primary_z = np.clip(primary_z, 0, max_z_idx)
        if is_multi_context or trust == 0:
            return [clamped_primary_z, clamped_primary_z, clamped_primary_z]
        else: # Single-motor context, trust > 0
            window_start = max(0, primary_z - trust)
            window_end = min(max_z_idx, primary_z + trust)
            possible_slices_in_window = list(range(window_start, window_end + 1))

            if len(possible_slices_in_window) < 3:
                s_minus_1 = np.clip(primary_z - 1, 0, max_z_idx)
                s_plus_1 = np.clip(primary_z + 1, 0, max_z_idx)
                return [s_minus_1, clamped_primary_z, s_plus_1]
            else:
                last_possible_start_idx_in_list = len(possible_slices_in_window) - 3
                chosen_start_offset_in_list = random.randint(0, last_possible_start_idx_in_list)
                s1 = possible_slices_in_window[chosen_start_offset_in_list]
                return [s1, s1 + 1, s1 + 2]

    def _get_negative_slice_indices(self, primary_z: int, max_z_idx: int, selection_method: str):
        clamped_primary_z = np.clip(primary_z, 0, max_z_idx)
        if selection_method == 'z_repeat_3':
            return [clamped_primary_z, clamped_primary_z, clamped_primary_z]
        elif selection_method == 'z_pm_1':
            s_minus_1 = np.clip(primary_z - 1, 0, max_z_idx)
            s_plus_1 = np.clip(primary_z + 1, 0, max_z_idx)
            return [s_minus_1, clamped_primary_z, s_plus_1]
        else:
            raise ValueError(f"Unknown slice_selection_negative: {selection_method}")

    def _load_slice(self, tomogram_id: str, slice_idx: int, tomo_shape_w: int, tomo_shape_h: int) -> Image.Image:
        slice_filename = self.slicename_format.format(slice_idx)
        slice_path = os.path.join(self.image_base_dir, tomogram_id, slice_filename)
        try:
            img = Image.open(slice_path).convert('L')
            if img.size != (tomo_shape_w, tomo_shape_h):
                 # This could be an issue if not handled by augmentations or expected.
                 # For now, proceed, assuming full slice is used.
                 pass
            return img
        except FileNotFoundError:
            return Image.new('L', (tomo_shape_w, tomo_shape_h), 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        tomogram_id = sample_info["tomogram_id"]
        primary_z = sample_info["primary_z_for_sampling"]
        tomo_shape_z, H_orig, W_orig = (sample_info["tomo_shape_z"], 
                                        sample_info["tomo_shape_h"], 
                                        sample_info["tomo_shape_w"])
        max_z_idx = tomo_shape_z - 1
        label_type = sample_info["label_type"]
        is_multi_context = sample_info.get("is_multi_motor_context", False)

        if label_type == 'positive':
            slice_indices = self._get_positive_slice_indices(primary_z, max_z_idx, is_multi_context, self.trust_param_positive)
        else: # Negative
            slice_indices = self._get_negative_slice_indices(primary_z, max_z_idx, self.slice_selection_negative)

        # 1. Load slices -> (C, H_orig, W_orig) numpy array, uint8, 0-255
        slices_pil = [self._load_slice(tomogram_id, s_idx, W_orig, H_orig) for s_idx in slice_indices]
        # Ensure slices are uint8 for Albumentations if it expects that
        stacked_slices_np_uint8 = np.stack([np.array(img_pil, dtype=np.uint8) for img_pil in slices_pil], axis=0)

        # 2. Prepare keypoints (GT centers) for spatial augmentation
        keypoints_for_aug = []
        keypoint_class_labels_dummy = [] # Dummy labels for keypoints if aug lib needs them
        if label_type == 'positive':
            for gt_center_obj in sample_info['gt_objects_centers_px']:
                keypoints_for_aug.append((gt_center_obj['x_center_px'], gt_center_obj['y_center_px']))
                keypoint_class_labels_dummy.append(gt_center_obj['class_id']) # This is the motor class_id (0)

        # 3. Apply Spatial Augmentations (if training mode)
        # Albumentations expects image as (H,W,C)
        img_HWC_uint8 = stacked_slices_np_uint8.transpose(1,2,0) 
        
        augmented_img_HWC = img_HWC_uint8
        augmented_keypoints_centers = keypoints_for_aug # List of (x,y) tuples

        if self.is_train and self.spatial_transform:
            # Pass keypoint_class_labels if your A.KeypointParams has label_fields defined
            aug_result = self.spatial_transform(image=img_HWC_uint8, 
                                                keypoints=keypoints_for_aug, 
                                                keypoint_class_labels=keypoint_class_labels_dummy)
            augmented_img_HWC = aug_result['image']
            augmented_keypoints_centers = aug_result['keypoints']
        
        # Transpose image back to (C,H,W)
        augmented_img_CHW_uint8 = augmented_img_HWC.transpose(2,0,1)
        H_aug, W_aug = augmented_img_CHW_uint8.shape[1], augmented_img_CHW_uint8.shape[2]

        # 4. Apply Color Augmentations (if training mode)
        # Color transforms might expect (H,W,C) or (C,H,W), uint8 or float.
        # Most Albumentations color transforms work on HWC, uint8.
        final_img_CHW_uint8 = augmented_img_CHW_uint8
        if self.is_train and self.color_transform:
            # If color_transform also expects HWC:
            temp_color_img_HWC = augmented_img_CHW_uint8.transpose(1,2,0)
            colored_img_HWC = self.color_transform(image=temp_color_img_HWC)['image']
            final_img_CHW_uint8 = colored_img_HWC.transpose(2,0,1)
            # Else if color_transform expects CHW, apply directly.

        # 5. Normalize image to float and desired range (0-1 or 0-255)
        final_img_CHW_float = final_img_CHW_uint8.astype(np.float32)
        if self.normalize_output_range: # Normalize to 0-1
            final_img_CHW_float /= 255.0
            final_img_CHW_float = np.clip(final_img_CHW_float, 0.0, 1.0)
        # else: it remains 0-255 float

        # 6. Convert augmented keypoint centers to YOLO bounding boxes
        yolo_labels_list = []
        if label_type == 'positive' and augmented_keypoints_centers:
            if H_aug == 0 or W_aug == 0: # Image augmented to zero size
                pass # yolo_labels_list remains empty
            else:
                for i, (kx_aug, ky_aug) in enumerate(augmented_keypoints_centers):
                    # class_id for YOLO is from original GT object (dummy keypoint label)
                    yolo_class_id = keypoint_class_labels_dummy[i] 

                    # Calculate x_min, y_min, x_max, y_max from augmented center and FIXED bb size
                    x_min_px = kx_aug - BB_WIDTH_PX / 2.0
                    y_min_px = ky_aug - BB_HEIGHT_PX / 2.0
                    x_max_px = kx_aug + BB_WIDTH_PX / 2.0
                    y_max_px = ky_aug + BB_HEIGHT_PX / 2.0

                    # Clip to augmented image boundaries
                    x_min_px = np.clip(x_min_px, 0, W_aug)
                    y_min_px = np.clip(y_min_px, 0, H_aug)
                    x_max_px = np.clip(x_max_px, 0, W_aug)
                    y_max_px = np.clip(y_max_px, 0, H_aug)
                    
                    # Filter out boxes that are too small or outside image after clipping
                    if x_max_px <= x_min_px + 1e-3 or y_max_px <= y_min_px + 1e-3: # Check for valid width/height
                        continue

                    # Convert to YOLO format (normalized by augmented image dimensions)
                    box_w_norm = (x_max_px - x_min_px) / W_aug
                    box_h_norm = (y_max_px - y_min_px) / H_aug
                    x_center_norm = (x_min_px + x_max_px) / 2.0 / W_aug
                    y_center_norm = (y_min_px + y_max_px) / 2.0 / H_aug
                    
                    yolo_labels_list.append([yolo_class_id, x_center_norm, y_center_norm, box_w_norm, box_h_norm])

        # 7. Convert to PyTorch Tensors
        output_image_tensor = torch.from_numpy(final_img_CHW_float)
        
        if yolo_labels_list:
            output_labels_tensor = torch.tensor(yolo_labels_list, dtype=torch.float32)
        else:
            output_labels_tensor = torch.empty((0, 5), dtype=torch.float32) # Standard for no objects
            
        return output_image_tensor, output_labels_tensor


# --- Example Usage (remember to define actual augmentation functions) ---
# aug_spatial_train = get_spatial_augmentations(is_train=True) # Your Albumentations compose
# aug_color_train = get_color_augmentations(is_train=True)     # Your Albumentations compose

# dummy_positive_df = pd.DataFrame({
#     'tomogram_id': ['tomo1', 'tomo1', 'tomo1', 'tomo2'],
#     'x': [100, 150, 200, 50],
#     'y': [100, 110, 100, 60],
#     'z': [10, 10, 15, 5],
#     'num_motors': [1, 1, 1, 1],
#     'tomo_shape_z': [30, 30, 30, 20],
#     'tomo_shape_h': [256, 256, 256, 256],
#     'tomo_shape_w': [256, 256, 256, 256]
# })
# dummy_negative_list = [{
#     'tomogram_id': 'tomo1', 'x': 50, 'y': 50, 'z': 5,
#     'tomo_shape_z': 30, 'tomo_shape_h': 256, 'tomo_shape_w': 256
# }]

# train_dataset = Tomo25DDataset(
#     positive_labels_df=dummy_positive_df,
#     image_base_dir="path/to/your/image_folders", # Replace with actual path
#     negative_coords_list=dummy_negative_list,
#     spatial_transform_fn=aug_spatial_train,
#     color_transform_fn=aug_color_train,
#     is_train=True,
#     normalize_output_range=True # Output tensor 0-1
# )

# if len(train_dataset) > 0:
#     img, lbls = train_dataset[0]
#     print("Image tensor shape:", img.shape)
#     print("Labels tensor shape:", lbls.shape)
#     print("Labels:", lbls)

#     # For DataLoader, you'll need a collate_fn that can handle lists of labels if batch_size > 1
#     # since the number of detected objects (N in N,5) can vary.
#     # PyTorch's default collate_fn will work if labels are lists of tensors,
#     # or if you always return a tensor (even if (0,5)).
#     # DataLoader will batch images if they are same size. If augmentations make them
#     # variable size, you'll need a more complex collate_fn to pad images.
# else:
#     print("Dataset is empty, check data or _prepare_samples.")
