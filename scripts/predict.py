#!/usr/bin/env python3
"""
End-to-end BYU-25 inference on a single tomogram.

Pipeline (per tomogram):
  - load each JPEG slice (``<tomo_dir>/<zz>.jpg``)
  - run Ultralytics YOLO inference per slice
  - collect detections as ``{z: [(xc, yc, w, h, cls, conf), ...]}``
  - feed that dict into :func:`byu.postproc.nms3d.merge_2d_predictions_to_3d`
  - emit the top-confidence 3D box as the motor location

The script is GPU-friendly (Ultralytics handles device placement), and honors the same
thresholds surface as the original notebook.

Usage:
    python scripts/predict.py \\
        --ckpt runs/detect/wt_dilated/weights/best.pt \\
        --tomo-dir data/test/tomo_xyz \\
        --out preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from byu.postproc.nms3d import merge_2d_predictions_to_3d  # noqa: E402


def _run_yolo_on_slices(ckpt: Path, slices: list[Path], conf: float, imgsz: int, device: str):
    """Return a list of per-slice lists of ``(xc, yc, w, h, cls, conf)``."""
    from ultralytics import YOLO  # lazy import - heavy dependency

    model = YOLO(str(ckpt))
    per_slice_raw = model.predict(
        source=[str(p) for p in slices],
        conf=conf,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )
    out: list[list[tuple[float, float, float, float, int, float]]] = []
    for res in per_slice_raw:
        boxes_for_slice: list[tuple[float, float, float, float, int, float]] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xywh = res.boxes.xywh.detach().cpu().numpy()
            clsses = res.boxes.cls.detach().cpu().numpy().astype(np.int64)
            confs = res.boxes.conf.detach().cpu().numpy()
            for (xc, yc, w, h), c, cf in zip(xywh, clsses, confs, strict=True):
                if w > 0 and h > 0:
                    boxes_for_slice.append((float(xc), float(yc), float(w), float(h), int(c), float(cf)))
        out.append(boxes_for_slice)
    return out


def _parse_z_from_path(path: Path) -> int:
    """Parse zero-padded z-index from filename like ``0123.jpg`` or ``slice_123.jpg``."""
    stem = path.stem
    if "_" in stem:
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            pass
    try:
        return int(stem)
    except ValueError as e:
        raise ValueError(f"Cannot parse z-index from {path!r}") from e


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="Trained YOLO checkpoint (.pt)")
    parser.add_argument("--tomo-dir", type=Path, required=True, help="Directory of .jpg slices")
    parser.add_argument("--out", type=Path, default=Path("preds.json"))
    parser.add_argument("--conf", type=float, default=0.30, help="2D detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iou-2d-link", type=float, default=0.30)
    parser.add_argument("--max-missed-slices", type=int, default=1)
    parser.add_argument("--min-slices-3d", type=int, default=3)
    parser.add_argument("--confidence-boost", type=float, default=0.20)
    parser.add_argument("--nms-iou-3d", type=float, default=0.20)
    parser.add_argument("--track-conf-thresh", type=float, default=0.47)
    args = parser.parse_args(argv)

    slices = sorted(args.tomo_dir.glob("*.jpg"))
    if not slices:
        print(f"No .jpg files found under {args.tomo_dir}")
        return 2
    print(f"Running {args.ckpt.name} over {len(slices)} slices from {args.tomo_dir}")

    per_slice_xywhc = _run_yolo_on_slices(args.ckpt, slices, args.conf, args.imgsz, args.device)
    per_slice_by_z: dict[int, list] = {}
    for path, dets in zip(slices, per_slice_xywhc, strict=True):
        try:
            z = _parse_z_from_path(path)
        except ValueError:
            continue
        per_slice_by_z[z] = dets

    boxes_3d = merge_2d_predictions_to_3d(
        per_slice_by_z,
        iou_threshold_2d_link=args.iou_2d_link,
        max_missed_slices=args.max_missed_slices,
        min_slices_for_3d=args.min_slices_3d,
        confidence_boost_factor=args.confidence_boost,
        nms_iou_threshold_3d=args.nms_iou_3d,
        track_confidence_threshold=args.track_conf_thresh,
    )

    payload = {
        "tomogram_dir": str(args.tomo_dir),
        "n_slices": len(slices),
        "detections_3d": [
            {
                "x1": b[0],
                "y1": b[1],
                "z1": b[2],
                "x2": b[3],
                "y2": b[4],
                "z2": b[5],
                "class": b[6],
                "confidence": b[7],
                "xc": (b[0] + b[3]) / 2,
                "yc": (b[1] + b[4]) / 2,
                "zc": (b[2] + b[5]) / 2,
            }
            for b in boxes_3d
        ],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {len(boxes_3d)} 3D detections to {args.out}")
    if boxes_3d:
        top = boxes_3d[0]
        print(
            f"Top pick: conf={top[7]:.3f} at (x={(top[0]+top[3])/2:.1f}, "
            f"y={(top[1]+top[4])/2:.1f}, z={(top[2]+top[5])/2:.1f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
