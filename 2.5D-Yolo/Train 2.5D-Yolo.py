import zipfile
import os
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from pathlib import Path
from tqdm.notebook import tqdm
from ultralytics import YOLO  # YOLO11 사용
import matplotlib.pyplot as plt
import torch
import random

# Set random seeds for reproducibility (paste.txt와 동일)
np.random.seed(2025)
random.seed(2025)
torch.manual_seed(2025)

# 데이터 경로 설정
yolo_dataset_dir = "/content/datasets/yolo_dataset_2_5d_byu"
yolo_images_dir = os.path.join(yolo_dataset_dir, "images")
yolo_labels_dir = os.path.join(yolo_dataset_dir, "labels")

# message.txt에서 생성된 2.5D 데이터 구조에 맞게 경로 설정
# message.txt 코드에서는 images/byu, labels/byu 구조를 사용
yolo_images_byu = os.path.join(yolo_dataset_dir, "images", "byu")
yolo_labels_byu = os.path.join(yolo_dataset_dir, "labels", "byu")

# 디렉토리 존재 확인
print(f"Images directory exists: {os.path.exists(yolo_images_byu)}")
print(f"Labels directory exists: {os.path.exists(yolo_labels_byu)}")

if os.path.exists(yolo_images_byu):
    print(f"Number of images: {len([f for f in os.listdir(yolo_images_byu) if f.endswith('.jpg')])}")
    
if os.path.exists(yolo_labels_byu):
    print(f"Number of labels: {len([f for f in os.listdir(yolo_labels_byu) if f.endswith('.txt')])}")

# 2.5D 스택 이미지 확인 및 변환 함수
def verify_and_convert_stack_images(images_dir):
    """
    2.5D 스택 이미지들이 3채널인지 확인하고, 필요시 그레이스케일을 3채널로 변환
    """
    converted_count = 0
    for img_file in tqdm(os.listdir(images_dir)):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(images_dir, img_file)
            img = Image.open(img_path)
            
            # 이미 3채널인 경우 (message.txt에서 생성된 2.5D 스택)
            if len(np.array(img).shape) == 3 and np.array(img).shape[2] == 3:
                continue
            
            # 그레이스케일인 경우 3채널로 변환
            elif len(np.array(img).shape) == 2 or (len(np.array(img).shape) == 3 and np.array(img).shape[2] == 1):
                img_array = np.array(img)
                if len(img_array.shape) == 2:
                    # 그레이스케일을 3채널로 복제
                    img_3ch = np.stack([img_array, img_array, img_array], axis=-1)
                else:
                    # 1채널을 3채널로 복제
                    img_3ch = np.repeat(img_array, 3, axis=-1)
                
                img_3ch_pil = Image.fromarray(img_3ch.astype(np.uint8))
                img_3ch_pil.save(img_path)
                converted_count += 1
                
    print(f"Converted {converted_count} images to 3-channel format")

# 이미지 변환 실행
if os.path.exists(yolo_images_byu):
    verify_and_convert_stack_images(yolo_images_byu)

# YAML 파일 생성 (message.txt에서 생성된 구조에 맞게)
yaml_content = {
    'path': str(Path(yolo_dataset_dir).resolve()),
    'train': 'train.txt',
    'val': 'val.txt',
    'nc': 1,
    'names': {0: 'motor'}
}

# 이미지 파일 리스트 생성 (byu 하위 디렉토리 구조 반영)
all_image_files_relative = []
if os.path.exists(yolo_images_byu):
    for f_name in os.listdir(yolo_images_byu):
        if f_name.endswith(".jpg"):
            all_image_files_relative.append(str(Path("images") / "byu" / f_name))
else:
    # 만약 byu 구조가 없다면 직접 images 디렉토리에서 가져오기
    for f_name in os.listdir(yolo_images_dir):
        if f_name.endswith(".jpg"):
            all_image_files_relative.append(str(Path("images") / f_name))

print(f"Total images found: {len(all_image_files_relative)}")

# Train/Val 분할
np.random.seed(42)
np.random.shuffle(all_image_files_relative)
split_idx = int(0.8 * len(all_image_files_relative))
train_files_relative = all_image_files_relative[:split_idx]
val_files_relative = all_image_files_relative[split_idx:]

print(f"Train images: {len(train_files_relative)}")
print(f"Val images: {len(val_files_relative)}")

# 파일 리스트 저장
yolo_dataset_path_obj = Path(yolo_dataset_dir).resolve()

with open(yolo_dataset_path_obj / 'train.txt', 'w') as f:
    for item_path_str in train_files_relative:
        f.write(f"./{item_path_str.replace(os.sep, '/')}\n")

with open(yolo_dataset_path_obj / 'val.txt', 'w') as f:
    for item_path_str in val_files_relative:
        f.write(f"./{item_path_str.replace(os.sep, '/')}\n")

# YAML 파일 저장
yaml_filename = 'dataset.yaml'
with open(os.path.join(yolo_dataset_dir, yaml_filename), 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

print(f"Dataset YAML created at: {os.path.join(yolo_dataset_dir, yaml_filename)}")

# YOLO11m 모델 로드 및 학습
model = YOLO('yolo11m.pt')  # YOLO11m 사용

# paste.txt에서 사용된 augmentation 설정 적용
augmentation_config = {
    'mosaic': 1.0,       # Mosaic augmentation (probability)
    'close_mosaic': 10,  # Close mosaic after specified epoch
    'mixup': 0.5,        # MixUp augmentation (probability)
    'flipud': 0.5,       # Flip up-down (probability)
    'scale': 0.25,       # Scale (+/- gain)
    'degrees': 45,       # Rotation (+/- deg)
}

# 학습 파라미터 설정 (paste.txt 기반으로 조정)
train_args = {
    "data": os.path.join(yolo_dataset_dir, yaml_filename),
    "epochs": 100,        # paste.txt와 동일하게 100 epochs
    "batch": 16,          # paste.txt와 동일하게 16
    "imgsz": 960,         # paste.txt와 동일하게 960
    "device": '0',
    "project": "/content",
    "name": "yolo11m_motor_detection",
    "exist_ok": True,
    
    # Optimizer settings (paste.txt 기반)
    "optimizer": "AdamW",
    "lr0": 1e-4,          # Initial learning rate (paste.txt와 동일)
    "lrf": 0.1,           # Final learning rate factor (paste.txt와 동일)
    "warmup_epochs": 0,   # No warmup epochs (paste.txt와 동일)
    "dropout": 0.25,      # Dropout rate (paste.txt와 동일)
    
    # Training management
    "patience": 100,      # Patience for early stopping
    "save_period": 5,     # Save checkpoints every 5 epochs (paste.txt와 동일)
    "val": True,          # Ensure validation is performed
    "verbose": True,      # Show detailed output during training
    
    # Apply augmentation config from paste.txt
    **augmentation_config
}

# 학습 시작
print("Starting YOLO11m training...")
results = model.train(**train_args)

# 학습 완료 후 loss curve 플롯
run_dir = os.path.join("/content", "yolo11m_motor_detection")
best_epoch_info = plot_dfl_loss_curve(run_dir)

if best_epoch_info:
    best_epoch, best_val_loss = best_epoch_info
    print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")

print("Training completed!")
def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for train and validation, marking the best model
    
    Args:
        run_dir (str): Directory where the training results are stored
    """
    # Path to the results CSV file
    results_csv = os.path.join(run_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    # Read results CSV
    results_df = pd.read_csv(results_csv)
    
    # Check if DFL loss columns exist
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    # Find the epoch with the best validation loss
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    
    # Mark the best model with a vertical line
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    
    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)
    plt.savefig('/content/dfl_loss_curve.png')
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    # Return the best epoch info
    return best_epoch, best_val_loss

# 학습 결과 확인 함수
def show_training_results():
    """학습 결과와 예측 샘플 확인"""
    import matplotlib.pyplot as plt
    
    # 검증 이미지로 예측 테스트
    val_images_dir = yolo_images_byu if os.path.exists(yolo_images_byu) else yolo_images_dir
    val_images = [f for f in os.listdir(val_images_dir) if f.endswith('.jpg')]
    
    if len(val_images) > 0:
        # 랜덤하게 4개 이미지 선택
        sample_images = np.random.choice(val_images, min(4, len(val_images)), replace=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for i, img_name in enumerate(sample_images):
            img_path = os.path.join(val_images_dir, img_name)
            
            # 예측 수행
            results = model.predict(img_path, conf=0.25, verbose=False)
            
            # 이미지 로드 및 표시
            img = Image.open(img_path)
            img_array = np.array(img)
            
            # 3채널 이미지인 경우 첫 번째 채널만 표시 (그레이스케일로)
            if len(img_array.shape) == 3:
                display_img = img_array[:, :, 0]
            else:
                display_img = img_array
                
            axes[i].imshow(display_img, cmap='gray')
            
            # 예측된 박스 그리기
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x1, y1, x2, y2 = box
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                    axes[i].add_patch(rect)
                    axes[i].text(x1, y1-5, f'{conf:.2f}', color='red', fontweight='bold')
            
            axes[i].set_title(f'{img_name}\nDetections: {len(results[0].boxes)}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/kaggle/working/prediction_samples.png', dpi=150, bbox_inches='tight')
        plt.show()

# 학습 완료 후 예측 샘플 확인
show_training_results()

# 모델 저장
model_save_path = "/kaggle/working/yolo11m_motor_best.pt"
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")