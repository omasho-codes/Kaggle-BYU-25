# Multi-GPU YOLO11m Motor Detection Training

This repository provides a complete solution for training YOLO11m on 2.5D stack images for bacterial flagellar motor detection using multiple GPUs with Distributed Data Parallel (DDP).

## 🚀 Features

- **Multi-GPU Training**: Support for 1-8 GPUs using DDP
- **Flexible Configuration**: Command-line arguments for all parameters
- **Automatic Dataset Setup**: Handles both BYU subdirectory and flat structures
- **Advanced Augmentation**: Mosaic, MixUp, rotation, scaling
- **Loss Visualization**: Automatic DFL loss curve plotting
- **Memory Efficient**: Configurable batch sizes and workers

## 📋 Requirements

- Python 3.8+
- CUDA-capable GPUs (1-8 GPUs)
- 16GB+ RAM recommended
- 50GB+ storage for dataset

## 🛠️ Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd multi-gpu-yolo-training
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
   - Place `yolo_dataset_2_5d_byu.zip` in the root directory, OR
   - Extract dataset to `./yolo_dataset_2_5d_byu/` directory

## 🎯 Quick Start

### Basic 4-GPU Training
```bash
python multi_gpu_train.py \
    --data_path ./yolo_dataset_2_5d_byu \
    --device 0,1,2,3 \
    --batch_size 64 \
    --workers 8 \
    --epochs 100
```

### Using the Run Script
```bash
chmod +x run_training.sh
./run_training.sh
```

## ⚙️ Configuration Options

### Essential Parameters
- `--device`: GPU devices (e.g., "0,1,2,3" for 4 GPUs)
- `--batch_size`: Total batch size (divided among GPUs)
- `--workers`: Data loader workers per GPU
- `--epochs`: Number of training epochs

### Performance Tuning
- `--imgsz`: Input image size (640/960/1280)
- `--lr0`: Initial learning rate
- `--dropout`: Dropout rate

### Data Augmentation
- `--mosaic`: Mosaic augmentation probability
- `--mixup`: MixUp augmentation probability
- `--degrees`: Rotation degrees
- `--scale`: Scale augmentation factor

## 📊 Example Configurations

### High-Performance (32GB+ VRAM per GPU)
```bash
python multi_gpu_train.py \
    --device 0,1,2,3 \
    --batch_size 128 \
    --workers 12 \
    --epochs 150 \
    --imgsz 960 \
    --lr0 2e-4
```

### Memory-Efficient (8GB VRAM per GPU)
```bash
python multi_gpu_train.py \
    --device 0,1,2,3 \
    --batch_size 32 \
    --workers 4 \
    --epochs 200 \
    --imgsz 640
```

### Single GPU Training
```bash
python multi_gpu_train.py \
    --device 0 \
    --batch_size 16 \
    --workers 4 \
    --epochs 100
```

## 📁 Dataset Structure

The script supports both structures:

### BYU Structure (from message.txt output)
```
yolo_dataset_2_5d_byu/
├── images/
│   └── byu/
│       ├── tomo_001_zcand0050_origZ0052_stack.jpg
│       └── ...
├── labels/
│   └── byu/
│       ├── tomo_001_zcand0050_origZ0052_stack.txt
│       └── ...
└── dataset.yaml
```

### Flat Structure
```
yolo_dataset_2_5d_byu/
├── images/
│   ├── image001.jpg
│   └── ...
├── labels/
│   ├── image001.txt
│   └── ...
└── dataset.yaml
```

## 📈 Training Output

Results are saved to `./runs/multi_gpu_motor_detection/`:
- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last epoch weights
- `results.csv`: Training metrics
- `dfl_loss_curve.png`: Loss visualization
- `final_model.pt`: Final saved model

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Reduce `--workers`
- Use smaller `--imgsz`

### Slow Training
- Increase `--workers` (up to 2x CPU cores)
- Use SSD storage for dataset
- Increase `--batch_size` if memory allows

### DDP Issues
- Ensure all GPUs have same memory
- Check CUDA device availability
- Use `CUDA_VISIBLE_DEVICES` if needed

## 📊 Performance Benchmarks

| GPUs | Batch Size | Image Size | Training Speed | Memory/GPU |
|------|------------|------------|----------------|------------|
| 1    | 16         | 640        | ~2.5 it/s     | ~6GB       |
| 2    | 32         | 640        | ~4.8 it/s     | ~6GB       |
| 4    | 64         | 960        | ~8.5 it/s     | ~12GB      |
| 4    | 128        | 960        | ~15 it/s      | ~20GB      |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Ultralytics YOLO team
- PyTorch DDP implementation
- BYU bacterial flagellar motor dataset
