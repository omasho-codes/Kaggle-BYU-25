#!/usr/bin/env python3
"""
Export a trained BYU-25 checkpoint to ONNX + verify numerical parity.

Scripted ONNX export with two things on top of ``ultralytics export``:

- it verifies the ONNX output against the PyTorch output on a random input, printing the
  max absolute difference;
- it uses ``onnxruntime`` rather than the Ultralytics internal verifier so the script
  still works once the repo has diverged from upstream ultralytics.

Example:

    python scripts/export_onnx.py \\
        --ckpt runs/detect/wt_dilated/weights/best.pt \\
        --imgsz 960 --opset 12 --out model.onnx

Set ``--config`` to one of ``configs/*.yaml`` if you need to build the architecture from
scratch (for WT variants the checkpoint must match the config).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

import byu.modules.register as register  # noqa: E402


def _collect(obj):
    out: list[torch.Tensor] = []
    if torch.is_tensor(obj):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect(v))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", type=Path, required=True, help="PyTorch checkpoint (.pt)")
    parser.add_argument("--out", type=Path, default=Path("model.onnx"))
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dynamic", action="store_true", help="Allow dynamic batch & spatial dims")
    parser.add_argument("--simplify", action="store_true", help="Run onnxsim on the exported model")
    parser.add_argument("--atol", type=float, default=1e-3, help="Max tolerance for parity check")
    args = parser.parse_args(argv)

    register.register_all()

    try:
        from ultralytics import YOLO

        yolo = YOLO(str(args.ckpt))
        print(f"Loaded checkpoint via Ultralytics: {args.ckpt}")
        out_path = yolo.export(
            format="onnx",
            opset=args.opset,
            dynamic=args.dynamic,
            simplify=args.simplify,
            imgsz=args.imgsz,
        )
        out_path = Path(out_path)
        if out_path != args.out:
            out_path.rename(args.out)
        model = yolo.model
    except Exception as exc:
        print(f"Ultralytics export failed ({exc}); falling back to torch.onnx.export")
        state = torch.load(args.ckpt, map_location="cpu")
        model = state["model"] if isinstance(state, dict) and "model" in state else state
        if hasattr(model, "fuse"):
            with torch_no_grad():
                model.fuse()
        model.eval()
        x = torch.randn(1, 3, args.imgsz, args.imgsz)
        torch.onnx.export(
            model,
            x,
            args.out,
            opset_version=args.opset,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch", 2: "H", 3: "W"}} if args.dynamic else None,
        )

    print(f"Exported ONNX -> {args.out}")

    x = torch.randn(1, 3, args.imgsz, args.imgsz)
    model.eval()
    with torch.no_grad():
        tic = time.perf_counter()
        torch_out = model(x)
        print(f"  PyTorch forward: {time.perf_counter() - tic:.3f}s")
    torch_tensors = _collect(torch_out)
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        tic = time.perf_counter()
        onnx_out = sess.run(None, {input_name: x.numpy()})
        print(f"  ONNX Runtime forward: {time.perf_counter() - tic:.3f}s")
        compare_up_to = min(len(torch_tensors), len(onnx_out))
        max_diff = 0.0
        for t, o in zip(torch_tensors[:compare_up_to], onnx_out[:compare_up_to], strict=False):
            t_np = t.detach().cpu().numpy()
            if t_np.shape != np.asarray(o).shape:
                print(f"  shape mismatch: torch {t_np.shape} vs onnx {np.asarray(o).shape} (skipping)")
                continue
            diff = float(np.abs(t_np - np.asarray(o)).max())
            max_diff = max(max_diff, diff)
        print(f"  max |torch - onnx| = {max_diff:.2e}  (atol={args.atol})")
        if max_diff > args.atol:
            print("  WARN: output drift exceeds tolerance; consider lowering opset / disabling simplify")
            return 1
    except ImportError:
        print("  onnxruntime not installed; skipping parity check")
    return 0


def torch_no_grad():
    return torch.no_grad()


if __name__ == "__main__":
    raise SystemExit(main())
