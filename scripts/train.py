#!/usr/bin/env python3
"""
Thin training launcher for the BYU-25 MHAF-YOLO + weight-tied clones ablation.

Registers our custom blocks, loads a config YAML via Ultralytics ``YOLO``, and kicks
off training. The three configs that matter:

- ``configs/mhaf_yolov2_n_baseline.yaml``    (baseline MHAF-YOLOv2-n)
- ``configs/mhaf_yolov2_n_wt_dilated.yaml``  (variant A: dilated weight-tied clones)
- ``configs/mhaf_yolov2_n_wt_gpyramid.yaml`` (variant C: Gaussian pyramid WT clones)

Single-GPU:
    python scripts/train.py --config configs/mhaf_yolov2_n_wt_dilated.yaml \\
        --data data/yolo_byu.yaml --epochs 100 --batch 16 --imgsz 960

Multi-GPU DDP:
    torchrun --nproc-per-node=4 scripts/train.py \\
        --config configs/mhaf_yolov2_n_wt_dilated.yaml \\
        --data data/yolo_byu.yaml --epochs 100 --batch 64 --imgsz 960 \\
        --device 0,1,2,3

The ``--data`` YAML follows the standard Ultralytics format (train/val image lists).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

import byu.modules.register as register  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="Model YAML (configs/*.yaml)")
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML (train/val paths)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("./runs/detect"))
    parser.add_argument("--name", default="byu25")
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    register.register_all()

    from ultralytics import YOLO

    model = YOLO(str(args.config), task="detect")
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        lr0=args.lr0,
        dropout=args.dropout,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
