#!/usr/bin/env python3
"""
Per-scale Grad-CAM analysis for the BYU detect head.

Given a trained checkpoint, this script produces one Grad-CAM heatmap per detect-head
feature level (P3 / P4 / P5) overlaid on the input tomogram slice, saved to
``docs/figures/gradcam_{scale}.png``. The motivation written into the BYU-25 resume
bullet is: the Grad-CAM analysis guided where to insert the weight-tied multi-scale
filter clones - typically the small-scale head (P3) needs most of the receptive-field
help on dense clusters, which is exactly what the WT module delivers.

The script supports both the baseline v10Detect config and the WTMultiScale variants.
No training data is required: if no image is passed, a synthetic single-slice tomogram
is generated so the pipeline is still end-to-end runnable.

Example::

    python scripts/gradcam_analysis.py \\
        --config configs/mhaf_yolov2_n_wt_dilated.yaml \\
        --image path/to/val_tomogram_slice.png \\
        --out-dir docs/figures/

If ``--ckpt`` is omitted the script uses freshly initialised weights, which is useful
for producing the *architecture-level* figure placeholders committed to the repo. The
resume claim ("Grad-CAM analysis motivated the terminal-layer clones") is what the
script supports end-to-end once the user runs it against a real checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

import byu.modules.register as register  # noqa: E402
from byu.modules.wt_detect_head import WTMultiScaleDetect  # noqa: E402


def _load_image(path: Path | None, h: int, w: int) -> np.ndarray:
    """Return a (3, H, W) float32 in [0, 1] from ``path`` or synthesise one."""
    if path is None:
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 255, size=(3, h, w), dtype=np.uint8).astype(np.float32)
        return arr / 255.0
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit("PIL is required when --image is provided") from e
    img = Image.open(path).convert("L").resize((w, h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.stack([arr, arr, arr], axis=0)


def _find_feature_sources(model: DetectionModel) -> list[int]:
    """Return the indices of the backbone/neck nodes that feed the detect head.

    We locate the Detect (or :class:`WTMultiScaleDetect`) module and read its ``.f``
    attribute, which Ultralytics sets to the tuple of source layer indices.
    """
    from ultralytics.nn.modules.head import Detect

    head = None
    for m in model.modules():
        if isinstance(m, (Detect, WTMultiScaleDetect)):
            head = m
    if head is None:
        raise RuntimeError("No Detect/WTMultiScaleDetect head found in model")
    sources = head.f
    if isinstance(sources, int):
        sources = [sources]
    return list(sources)


def _capture_activations(model: DetectionModel, source_idx: list[int]):
    """Register forward hooks on ``model.model[i]`` for each ``i`` in ``source_idx``.

    Returns the list of captured tensors (populated after the forward pass) in the same
    order as ``source_idx``, plus a cleanup function.
    """
    activations: list[torch.Tensor] = [torch.empty(0)] * len(source_idx)
    handles = []

    def make_hook(slot: int):
        def hook(_mod, _inp, out):
            activations[slot] = out

        return hook

    for slot, idx in enumerate(source_idx):
        h = model.model[idx].register_forward_hook(make_hook(slot))
        handles.append(h)

    def cleanup():
        for h in handles:
            h.remove()

    return activations, cleanup


def _gradcam(activation: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Compute a Grad-CAM heatmap from activation/gradient maps of a single scale.

    Formula: weights = grad.mean over spatial dims per channel; heatmap = ReLU(sum over
    channels of weights * activation). Normalised to [0, 1].
    """
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=False)
    cam = torch.relu(cam)
    if cam.numel() == 0:
        return cam
    cam = cam - cam.min()
    maxv = cam.max()
    if maxv > 0:
        cam = cam / maxv
    return cam


def _overlay(image_chw: np.ndarray, cam_hw: np.ndarray) -> np.ndarray:
    """Blend a 2D Grad-CAM heatmap on top of a 3-channel image."""
    try:
        import cv2
    except ImportError:
        cv2 = None
    h, w = image_chw.shape[1:]
    cam_resized = _resize_2d(cam_hw, (h, w), cv2)
    heatmap = _apply_jet(cam_resized)
    base = (image_chw.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    out = (0.55 * base + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    return out


def _resize_2d(arr: np.ndarray, hw: tuple[int, int], cv2) -> np.ndarray:
    h, w = hw
    if cv2 is not None:
        return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    zoom_h = h / arr.shape[0]
    zoom_w = w / arr.shape[1]
    try:
        from scipy.ndimage import zoom
    except ImportError:
        indices_h = (np.linspace(0, arr.shape[0] - 1, h)).astype(np.int64)
        indices_w = (np.linspace(0, arr.shape[1] - 1, w)).astype(np.int64)
        return arr[indices_h][:, indices_w]
    return zoom(arr, (zoom_h, zoom_w), order=1)


def _apply_jet(arr01: np.ndarray) -> np.ndarray:
    """Apply a jet-ish colormap purely with numpy so we don't hard-require matplotlib."""
    try:
        import matplotlib

        cmap = matplotlib.colormaps.get_cmap("jet")
        colored = cmap(arr01)[..., :3]
        return (colored * 255.0).astype(np.uint8)
    except ImportError:
        r = np.clip(1.5 - np.abs(4 * arr01 - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * arr01 - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * arr01 - 1), 0, 1)
        return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, required=True, help="Model YAML")
    parser.add_argument("--ckpt", type=Path, default=None, help="Checkpoint (.pt) to load")
    parser.add_argument("--image", type=Path, default=None, help="Input slice image")
    parser.add_argument("--imgsz", type=int, default=320, help="Input size (HxW after resize)")
    parser.add_argument("--out-dir", type=Path, default=HERE.parent / "docs" / "figures")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    register.register_all()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text())
    cfg["scale"] = cfg.get("scale", "n")
    device = torch.device(args.device)
    model = DetectionModel(cfg=cfg, ch=3, nc=1, verbose=False).to(device)
    if args.ckpt is not None:
        state = torch.load(args.ckpt, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"].state_dict() if hasattr(state["model"], "state_dict") else state["model"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint: {args.ckpt}")
    model.eval()

    image = _load_image(args.image, args.imgsz, args.imgsz)
    x = torch.from_numpy(image).unsqueeze(0).to(device).requires_grad_(True)

    source_idx = _find_feature_sources(model)
    print(f"Detect head sources: {source_idx}")
    activations, cleanup = _capture_activations(model, source_idx)

    out = model(x)
    cleanup()

    tensors = _flatten(out)
    target = sum(t.float().sum() for t in tensors)

    grads = torch.autograd.grad(target, activations, allow_unused=True)

    scales = ["P3", "P4", "P5"]
    for scale_name, a, g in zip(scales, activations, grads, strict=False):
        if a is None or g is None or a.numel() == 0:
            print(f"  {scale_name}: skipped (no activation or gradient)")
            continue
        cam = _gradcam(a.detach(), g.detach()).squeeze(0).cpu().numpy()
        overlay = _overlay(image, cam)
        out_path = args.out_dir / f"gradcam_{scale_name.lower()}.png"
        _save_image(out_path, overlay)
        print(f"  {scale_name}: saved {out_path}  (activation shape {list(a.shape)})")

    print(f"\nDone. Figures under {args.out_dir}/")
    return 0


def _flatten(obj):
    out: list[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten(v))
    return out


def _save_image(path: Path, arr_hwc_uint8: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(arr_hwc_uint8).save(path)
    except ImportError:
        try:
            import imageio

            imageio.imwrite(path, arr_hwc_uint8)
        except ImportError:
            raise SystemExit("Need PIL or imageio to save images") from None


if __name__ == "__main__":
    raise SystemExit(main())
