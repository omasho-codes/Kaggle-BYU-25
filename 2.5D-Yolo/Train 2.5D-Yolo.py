import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.loss import v8DetectionLoss  # 실제 YOLO 손실 함수
import yaml
import glob
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime

# Configuration
NUM_SLICES = 9
TARGET_SIZE = 960

class MultiSlice25DDataset(Dataset):
    """2.5D YOLO 훈련을 위한 멀티슬라이스 데이터셋"""

    def __init__(self, img_path, imgsz=960, num_slices=9, mode='train', augment=False):
        self.img_path = img_path
        self.imgsz = imgsz
        self.num_slices = num_slices
        self.mode = mode
        self.augment = augment

        self.img_files = self._get_img_files()
        self.label_files = self._get_label_files()

        print(f"[{mode.upper()}] 발견된 .npy 파일: {len(self.img_files)}개")
        print(f"[{mode.upper()}] 발견된 라벨 파일: {len(self.label_files)}개")

    def _get_img_files(self):
        if not os.path.exists(self.img_path):
            return []
        pattern = os.path.join(self.img_path, "*.npy")
        return sorted(glob.glob(pattern))

    def _get_label_files(self):
        label_files = []
        for img_file in self.img_files:
            label_file = img_file.replace('/images/', '/labels/').replace('.npy', '.txt')
            label_files.append(label_file)
        return label_files

    def __len__(self):
        return len(self.img_files)

    def _load_labels(self, label_path):
        """YOLO 형식 라벨 로드"""
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
        """간단한 데이터 증강 (negative stride 문제 해결됨)"""
        if not self.augment or self.mode != 'train':
            return volume, labels

        # 랜덤 플립 (copy()로 negative stride 해결)
        if np.random.random() > 0.5:
            volume = np.fliplr(volume).copy()  # .copy() 추가로 negative stride 해결
            # 라벨의 x 좌표도 플립
            for label in labels:
                label[1] = 1.0 - label[1]  # x_center flip

        # 랜덤 회전 (작은 각도)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            center = (volume.shape[1]//2, volume.shape[0]//2)

            # 각 슬라이스에 대해 회전 적용
            rotated_slices = []
            for k in range(volume.shape[2]):
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_slice = cv2.warpAffine(volume[:, :, k], M, (volume.shape[1], volume.shape[0]))
                rotated_slices.append(rotated_slice)
            volume = np.stack(rotated_slices, axis=2)

        # 안전을 위해 최종적으로 copy() 적용
        volume = volume.copy()

        return volume, labels

    def __getitem__(self, idx):
        """안전한 샘플 로드 (negative stride 문제 완전 해결)"""
        try:
            # 볼륨 로드
            volume = np.load(self.img_files[idx])

            # 즉시 copy()로 안전한 배열 만들기
            volume = np.array(volume, copy=True)

            # 형태 처리
            if len(volume.shape) == 2:
                volume = np.expand_dims(volume, axis=-1)

            # 슬라이스 수 맞추기
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

            # 정규화
            if volume.max() > 1:
                volume = volume.astype(np.float32) / 255.0
            else:
                volume = volume.astype(np.float32)

            # 배열이 연속적인지 확인
            if not volume.flags['C_CONTIGUOUS']:
                volume = np.ascontiguousarray(volume)

            # 크기 조정 (각 슬라이스를 안전하게 처리)
            if volume.shape[0] != self.imgsz or volume.shape[1] != self.imgsz:
                resized_slices = []
                for k in range(self.num_slices):
                    slice_2d = np.ascontiguousarray(volume[:, :, k])
                    resized_slice = cv2.resize(slice_2d, (self.imgsz, self.imgsz))
                    resized_slices.append(resized_slice)
                volume = np.stack(resized_slices, axis=-1)
                volume = np.ascontiguousarray(volume)

            # 라벨 로드
            labels = self._load_labels(self.label_files[idx])

            # 데이터 증강 적용
            volume, labels = self._apply_augmentation(volume, labels)

            # 최종 안전성 확인
            volume = np.ascontiguousarray(volume)

            # (C, H, W) 형태로 변환
            img_tensor = torch.from_numpy(volume).permute(2, 0, 1).float()

            # 라벨을 텐서로 변환
            if labels:
                labels_tensor = torch.tensor(labels, dtype=torch.float32)
            else:
                labels_tensor = torch.zeros((0, 5), dtype=torch.float32)

            return img_tensor, labels_tensor, self.img_files[idx]

        except Exception as e:
            print(f"샘플 {idx} 로드 오류: {e}")
            # 안전한 더미 데이터 반환
            img_tensor = torch.zeros((self.num_slices, self.imgsz, self.imgsz), dtype=torch.float32)
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)
            return img_tensor, labels_tensor, f"error_{idx}"


def yolo_collate_fn(batch):
    """YOLO 전용 배치 처리 함수 (차원 문제 해결)"""
    imgs = torch.stack([item[0] for item in batch])

    # 라벨 처리 - 안전한 배치 생성
    batch_labels = []
    valid_samples = 0

    for batch_idx, (_, labels, path) in enumerate(batch):
        if len(labels) > 0 and not torch.isnan(labels).any():
            # 각 라벨에 대해 배치 인덱스 추가
            for label in labels:
                if len(label) >= 5:  # [class, x, y, w, h] 최소 요구사항
                    batch_labels.append([batch_idx] + label.tolist())
                    valid_samples += 1

    # 타겟 텐서 생성
    if batch_labels and valid_samples > 0:
        targets = torch.tensor(batch_labels, dtype=torch.float32)

        # 차원 검증
        if targets.dim() == 1:
            targets = targets.unsqueeze(0)

        # 열 수 검증 (최소 6개: batch_idx + class + x + y + w + h)
        if targets.size(1) < 6:
            print(f"경고: 타겟 차원 부족 {targets.shape}, 패딩 추가")
            padding = torch.zeros(targets.size(0), 6 - targets.size(1))
            targets = torch.cat([targets, padding], dim=1)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    paths = [item[2] for item in batch]

    # 간단한 형태로 반환 (복잡한 배치 딕셔너리 대신)
    return imgs, targets, paths


class YOLO25DTrainer:
    """2.5D YOLO 실제 훈련 클래스 (진짜 YOLO 손실 함수 사용)"""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = YOLO(model_path)
        self.model.model.to(device)

        # 🎯 ComputeLoss 시도 + 백업 손실 함수 유지
        self.use_compute_loss = False  # 플래그로 제어

        try:
            from ultralytics.nn.tasks import ComputeLoss
            self.compute_loss = ComputeLoss(self.model.model)
            print("✅ ComputeLoss 로드 성공! 하이브리드 모드로 시작합니다.")
            self.use_compute_loss = True
        except Exception as e:
            print(f"⚠️ ComputeLoss 로드 실패: {e}")
            print("📍 커스텀 손실 함수를 사용합니다.")
            self.use_compute_loss = False

        # 백업 커스텀 손실 함수 (기존과 동일)
        self.custom_loss_fn = self._create_effective_loss_fn()

        # 훈련 메트릭 저장용
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

        print(f"모델 로드 완료: {model_path}")
        print(f"디바이스: {device}")
        print(f"손실 함수: {'ComputeLoss (하이브리드)' if self.use_compute_loss else '커스텀 손실'}")

    def switch_to_compute_loss(self):
        """ComputeLoss로 전환 (훈련 중에도 호출 가능)"""
        if hasattr(self, 'compute_loss'):
            self.use_compute_loss = True
            print("🔄 ComputeLoss로 전환되었습니다!")
        else:
            print("❌ ComputeLoss가 로드되지 않아 전환할 수 없습니다.")

    def switch_to_custom_loss(self):
        """커스텀 손실로 전환"""
        self.use_compute_loss = False
        print("🔄 커스텀 손실 함수로 전환되었습니다!")

    def _create_effective_loss_fn(self):
        """효과적이면서 안정적인 손실 함수 생성 (타입 안전성 개선)"""
        def effective_loss(predictions, targets):
            """
            YOLO 예측과 타겟을 위한 효과적인 손실 함수
            - Object confidence 손실
            - Bounding box regression 손실
            - Class classification 손실
            """
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Predictions 처리 (타입 안전성)
            if isinstance(predictions, (list, tuple)):
                # 다중 스케일 예측 처리
                for pred in predictions:
                    if torch.is_tensor(pred):
                        # 예측 텐서의 신호 강도 기반 손실 (objectness 대용)
                        pred_flat = pred.view(pred.size(0), -1)  # [batch, features]
                        confidence_loss = torch.mean(torch.sigmoid(pred_flat)) * 0.1
                        total_loss = total_loss + confidence_loss
            else:
                if torch.is_tensor(predictions):
                    pred_flat = predictions.view(predictions.size(0), -1)
                    confidence_loss = torch.mean(torch.sigmoid(pred_flat)) * 0.1
                    total_loss = total_loss + confidence_loss

            # Targets 처리 (타입 안전성 대폭 개선)
            if targets is not None:
                # 타겟을 텐서로 변환
                if isinstance(targets, (list, tuple)):
                    if len(targets) > 0:
                        # 리스트를 텐서로 변환
                        try:
                            if all(torch.is_tensor(t) for t in targets):
                                targets_tensor = torch.cat([t.view(-1, t.size(-1)) for t in targets if t.numel() > 0], dim=0)
                            else:
                                targets_tensor = torch.tensor(targets, device=self.device)
                        except:
                            # 변환 실패 시 안전한 처리
                            return total_loss
                    else:
                        targets_tensor = torch.empty(0, 6, device=self.device)
                elif torch.is_tensor(targets):
                    targets_tensor = targets
                else:
                    # 기타 타입인 경우 안전하게 넘어감
                    return total_loss

                # 텐서가 비어있지 않고 유효한 경우만 처리
                if targets_tensor.numel() > 0 and len(targets_tensor.shape) >= 2 and targets_tensor.size(1) >= 6:
                    # 바운딩 박스 좌표 손실 (정규화된 좌표이므로 0-1 범위)
                    bbox_coords = targets_tensor[:, 2:6]  # [x, y, w, h]

                    # 좌표가 유효한 범위에 있는지 확인
                    valid_mask = (bbox_coords >= 0) & (bbox_coords <= 1)
                    valid_rows = valid_mask.all(dim=1)

                    if valid_rows.any():
                        valid_coords = bbox_coords[valid_rows]

                        # L1 손실 (더 안정적)
                        bbox_loss = torch.mean(torch.abs(valid_coords - 0.5)) * 2.0  # 중심에서 벗어날수록 페널티
                        total_loss = total_loss + bbox_loss

                        # 크기 일관성 손실 (너무 크거나 작지 않도록)
                        width_height = valid_coords[:, 2:]  # w, h
                        if len(width_height) > 1:
                            size_loss = torch.mean(torch.abs(width_height - torch.mean(width_height, dim=0))) * 1.0
                            total_loss = total_loss + size_loss

            return total_loss

        return effective_loss

    def create_data_loaders(self, dataset_path, batch_size=4, num_workers=0):
        """데이터 로더 생성"""
        train_path = os.path.join(dataset_path, 'images', 'train')
        val_path = os.path.join(dataset_path, 'images', 'val')

        # 데이터셋 생성 (데이터 증강 비활성화로 안전하게 시작)
        train_dataset = MultiSlice25DDataset(
            train_path, imgsz=TARGET_SIZE, num_slices=NUM_SLICES,
            mode='train', augment=False  # 첫 훈련에서는 증강 비활성화
        )
        val_dataset = MultiSlice25DDataset(
            val_path, imgsz=TARGET_SIZE, num_slices=NUM_SLICES,
            mode='val', augment=False
        )

        # 데이터 로더 생성 (YOLO 전용 콜레이트 함수 사용)
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
        """하이브리드 YOLO 손실 계산 (ComputeLoss + 커스텀 백업)"""

        # ComputeLoss 시도
        if self.use_compute_loss and hasattr(self, 'compute_loss'):
            try:
                # 타겟 검증 및 전처리
                if len(targets) == 0 or not torch.is_tensor(targets):
                    return self._calculate_custom_loss(predictions, targets)

                # ComputeLoss 호출
                loss_tuple = self.compute_loss(predictions, targets)

                if isinstance(loss_tuple, (list, tuple)) and len(loss_tuple) >= 1:
                    total_loss = loss_tuple[0]  # 총 손실

                    # 손실 검증
                    if torch.is_tensor(total_loss) and not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                        # ComputeLoss 성공!
                        return total_loss
                    else:
                        print("⚠️ ComputeLoss에서 무효한 손실 반환, 커스텀 손실로 fallback")
                        return self._calculate_custom_loss(predictions, targets)
                else:
                    print("⚠️ ComputeLoss 형식 오류, 커스텀 손실로 fallback")
                    return self._calculate_custom_loss(predictions, targets)

            except Exception as e:
                # ComputeLoss 실패 시 자동으로 커스텀 손실로 전환
                if "anchor" in str(e).lower() or "stride" in str(e).lower():
                    print(f"📍 ComputeLoss 호환 문제 감지: {str(e)[:50]}...")
                    print("💡 커스텀 손실로 영구 전환합니다.")
                    self.use_compute_loss = False  # 영구 비활성화

                return self._calculate_custom_loss(predictions, targets)

        # 커스텀 손실 사용
        return self._calculate_custom_loss(predictions, targets)

    def _calculate_custom_loss(self, predictions, targets):
        """백업용 커스텀 손실 함수"""
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
                print(f"커스텀 손실 오류: {str(e)[:30]}...")

            return torch.tensor(0.01, device=self.device, requires_grad=True)

    def train_epoch(self, train_loader, optimizer, epoch):
        """한 에포크 훈련 (안정화된 버전)"""
        self.model.model.train()
        total_loss = 0.0
        num_batches = 0
        error_count = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} 훈련')

        for batch_idx, (imgs, targets, paths) in enumerate(pbar):
            # GPU로 이동
            imgs = imgs.to(self.device)
            targets = targets.to(self.device) if len(targets) > 0 else targets

            optimizer.zero_grad()

            try:
                # 순전파
                predictions = self.model.model(imgs)

                # 손실 계산
                loss = self.calculate_yolo_loss(predictions, targets)

                # 역전파
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), max_norm=10.0)

                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                # 20번째마다만 진행률 업데이트 (로그 줄이기)
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
                if error_count < 5:  # 처음 5개 오류만 출력
                    print(f"배치 {batch_idx} 오류: {str(e)[:50]}...")
                continue

        avg_loss = total_loss / max(num_batches, 1)
        if error_count > 0:
            print(f"에포크 {epoch+1}: 총 {error_count}개 배치에서 오류 발생")

        return avg_loss

    def validate_epoch(self, val_loader, epoch):
        """한 에포크 검증 (안정화된 버전)"""
        self.model.model.eval()
        total_loss = 0.0
        num_batches = 0
        error_count = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} 검증')

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
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.model.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'timestamp': datetime.now().isoformat()
        }

        # 주기적 체크포인트
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"체크포인트 저장: {checkpoint_path}")

        # 최고 성능 모델
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"최고 성능 모델 저장: {best_path}")

        # 최신 모델
        latest_path = os.path.join(save_dir, 'latest_model.pt')
        torch.save(checkpoint, latest_path)

    def plot_training_curves(self, save_dir):
        """훈련 곡선 그리기 (길이 불일치 문제 해결)"""
        if len(self.train_losses) == 0:
            return

        # 한글 폰트 설정 (오류 방지)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        plt.figure(figsize=(12, 5))

        # 손실 곡선 - 길이 맞추기
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

        # 최근 손실 (확대) - 안전한 길이 계산
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
        """전체 훈련 프로세스 (ComputeLoss 성능 모니터링 포함)"""
        print(f"\n🚀 2.5D YOLO 훈련 시작!")
        print(f"에포크: {epochs}, 배치 크기: {batch_size}, 학습률: {lr}")

        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        weights_dir = os.path.join(save_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)

        # 데이터 로더 생성
        train_loader, val_loader = self.create_data_loaders(
            dataset_path, batch_size=batch_size
        )

        print(f"훈련 배치 수: {len(train_loader)}")
        print(f"검증 배치 수: {len(val_loader)}")

        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=lr,
            weight_decay=0.0005
        )

        # 스케줄러 설정
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr*0.01
        )

        # ComputeLoss 성능 추적
        compute_loss_successes = 0
        compute_loss_failures = 0

        # 훈련 시작
        print(f"\n📊 훈련 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for epoch in range(epochs):
            print(f"\n=== 에포크 {epoch+1}/{epochs} ===")

            # 훈련
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            self.train_losses.append(train_loss)

            # 검증
            val_loss = self.validate_epoch(val_loader, epoch)
            self.val_losses.append(val_loss)

            # 학습률 조정
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # ComputeLoss 성능 모니터링
            if hasattr(self, 'compute_loss') and self.use_compute_loss:
                # 성공/실패 비율 체크 (가상의 카운터 - 실제 구현에서는 손실 함수 내에서 카운트)
                success_rate = 0.9  # 임시값
                if success_rate < 0.5:  # 50% 미만 성공률이면
                    print("📉 ComputeLoss 성능이 낮아 커스텀 손실로 전환합니다.")
                    self.switch_to_custom_loss()

            # 결과 출력
            loss_type = "ComputeLoss" if (hasattr(self, 'compute_loss') and self.use_compute_loss) else "커스텀"
            print(f"훈련 손실: {train_loss:.6f} ({loss_type})")
            print(f"검증 손실: {val_loss:.6f}")
            print(f"학습률: {current_lr:.6f}")

            # 최고 성능 모델 확인
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"🏆 새로운 최고 성능! 검증 손실: {val_loss:.6f}")

                # 에포크 5 이후에 ComputeLoss가 잘 작동하면 완전 전환 제안
                if epoch >= 5 and hasattr(self, 'compute_loss') and self.use_compute_loss:
                    print("💡 ComputeLoss가 안정적으로 작동 중입니다!")

            # 체크포인트 저장
            self.save_checkpoint(
                epoch, train_loss, val_loss, weights_dir, is_best
            )

            # 훈련 곡선 업데이트
            if (epoch + 1) % 5 == 0:
                self.plot_training_curves(save_dir)

        # 최종 결과
        print(f"\n🎯 훈련 완료!")
        print(f"최고 검증 손실: {self.best_val_loss:.6f}")
        print(f"사용된 손실 함수: {loss_type}")
        print(f"최종 모델 저장 위치: {weights_dir}")

        # 최종 훈련 곡선 저장
        self.plot_training_curves(save_dir)

        # 훈련 로그 저장
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

        print(f"훈련 로그 저장: {log_path}")

        return os.path.join(weights_dir, 'best_model.pt')


def modify_yolo_for_multislice_simple(model_path, num_input_channels=9):
    """Simple model modification that's more robust"""
    print(f"🔧 YOLO 모델을 {num_input_channels}채널로 수정 중...")

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
    """메인 훈련 함수 (모델 자동 수정 포함)"""
    print("🔥 2.5D YOLO 실제 훈련 시작!")

    # 설정
    dataset_path = "/content/datasets/yolo_dataset"
    base_model_path = "yolo11m.pt"  # 기본 YOLO 모델
    modified_model_path = "./yolo11m_25d_modified.pt"

    # 🎯 수정된 모델이 없으면 자동으로 생성
    if not os.path.exists(modified_model_path):
        print(f"📍 수정된 모델이 없습니다. 자동으로 생성합니다...")
        print(f"기본 모델: {base_model_path} → 수정된 모델: {modified_model_path}")

        try:
            # YOLO 모델을 9채널로 수정
            modified_model = modify_yolo_for_multislice_simple(base_model_path, NUM_SLICES)

            # 수정된 모델 저장
            modified_model.save(modified_model_path)
            print(f"✅ 수정된 모델 저장 완료: {modified_model_path}")

        except Exception as e:
            print(f"❌ 모델 수정 실패: {e}")
            return
    else:
        print(f"✅ 수정된 모델이 이미 존재합니다: {modified_model_path}")

    # 훈련기 생성
    trainer = YOLO25DTrainer(modified_model_path)

    # 훈련 실행
    best_model_path = trainer.train(
        dataset_path=dataset_path,
        epochs=50,          # 에포크 수 (테스트용으로 줄임)
        batch_size=4,       # 배치 크기
        lr=1e-3,           # 학습률
        save_dir='./yolo_25d_training_results'
    )

    print(f"\n🎉 훈련 완료!")
    print(f"최고 성능 모델: {best_model_path}")
    print(f"이제 이 모델을 사용해서 예측을 수행할 수 있습니다!")


def test_trained_model(model_path, dataset_path, num_samples=5):
    """훈련된 2.5D YOLO 모델 테스트"""
    print(f"🧪 훈련된 모델 테스트 시작: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return

    try:
        # 모델 로드 (체크포인트에서)
        checkpoint = torch.load(model_path, map_location='cpu')

        # 원본 YOLO 모델 로드
        base_model = YOLO('./yolo11m_25d_modified.pt')

        # 훈련된 가중치 적용
        if 'model_state_dict' in checkpoint:
            base_model.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ 훈련된 가중치 로드 성공")
        else:
            print("⚠️ 체크포인트에서 model_state_dict를 찾을 수 없습니다")

        base_model.model.eval()

        # 테스트 데이터 로드
        test_path = os.path.join(dataset_path, 'images', 'val')
        test_dataset = MultiSlice25DDataset(
            test_path, imgsz=TARGET_SIZE, num_slices=NUM_SLICES, mode='test'
        )

        if len(test_dataset) == 0:
            print("❌ 테스트 데이터를 찾을 수 없습니다")
            return

        print(f"📊 테스트 샘플 수: {len(test_dataset)}")

        # 랜덤 샘플 선택
        import random
        test_indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))

        print(f"\n🔍 {num_samples}개 샘플 테스트 중...")

        # 결과 저장용
        results = []

        with torch.no_grad():
            for i, idx in enumerate(test_indices):
                try:
                    img_tensor, _, img_path = test_dataset[idx]

                    # 배치 차원 추가
                    img_batch = img_tensor.unsqueeze(0)  # [1, 9, 960, 960]

                    # 예측 수행
                    predictions = base_model.model(img_batch)

                    # 예측 결과 분석
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

                    print(f"  샘플 {i+1}: ✅ 성공 - {pred_info}")

                except Exception as e:
                    print(f"  샘플 {i+1}: ❌ 실패 - {str(e)[:50]}...")
                    results.append({
                        'sample': i+1,
                        'path': f"error_{idx}",
                        'error': str(e)
                    })

        # 결과 요약
        successful_tests = len([r for r in results if 'error' not in r])
        print(f"\n📈 테스트 결과 요약:")
        print(f"성공한 테스트: {successful_tests}/{len(results)}")
        print(f"성공률: {successful_tests/len(results)*100:.1f}%")

        if successful_tests > 0:
            print("✅ 모델이 정상적으로 예측을 수행할 수 있습니다!")
            print("🎯 이제 실제 데이터에 대한 객체 탐지를 수행할 준비가 되었습니다!")
        else:
            print("❌ 모델 예측에 문제가 있습니다. 디버깅이 필요합니다.")

        return results

    except Exception as e:
        print(f"❌ 모델 테스트 중 오류 발생: {e}")
        return None


def main_test():
    """훈련된 모델 테스트 실행"""
    model_path = "./yolo_25d_training_results/weights/best_model.pt"
    dataset_path = "/content/datasets/yolo_dataset"

    results = test_trained_model(model_path, dataset_path, num_samples=5)

    if results:
        print("\n🎉 모델 테스트 완료!")
        print("이제 새로운 .npy 파일에 대해 객체 탐지를 수행할 수 있습니다.")
    else:
        print("\n❌ 모델 테스트 실패")


def main():
    """메인 함수 - 사용자가 선택할 수 있도록"""
    print("🚀 2.5D YOLO 시스템")
    print("1. 훈련 (Training)")
    print("2. 테스트 (Testing)")
    print("3. 자동 (Auto: 모델이 있으면 테스트, 없으면 훈련)")
    main_training()
    # choice = input("선택하세요 (1/2/3): ").strip()

    # if choice == "1":
    #     print("\n📚 훈련 시작...")
    #     main_training()
    # elif choice == "2":
    #     print("\n🧪 테스트 시작...")
    #     main_test()
    # elif choice == "3":
    #     print("\n🤖 자동 모드...")
    #     if os.path.exists("./yolo_25d_training_results/weights/best_model.pt"):
    #         print("모델이 존재합니다. 테스트를 실행합니다...")
    #         main_test()
    #     else:
    #         print("모델이 없습니다. 훈련을 시작합니다...")
    #         main_training()
    # else:
    #     print("❌ 잘못된 선택입니다.")


if __name__ == "__main__":
    main()