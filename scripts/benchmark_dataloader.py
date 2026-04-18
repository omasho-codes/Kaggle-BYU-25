#!/usr/bin/env python3
"""
Benchmark: ThreadedDataLoader vs DataLoader(num_workers=N) vs DataLoader(num_workers=0).

Runs each config ``--n-passes`` times over a dataset materialised from ``--n-tomos``
tomograms and emits:

- ``benchmarks/dataloader_results.md``   : markdown table
- ``benchmarks/dataloader_results.json`` : raw numbers per config / per run
- ``benchmarks/dataloader_results.png``  : bar chart (total wall time, mean)

By default the script materialises synthetic tomograms under ``--data-root`` so the
benchmark is reproducible even without the full BYU dataset. Point ``--data-root`` at
a real ``yolo_dataset_2_5d_byu`` directory to measure realistic disk latency.

Example:

    python scripts/benchmark_dataloader.py --n-tomos 3 --n-passes 3

Typical invocation on a single-GPU box with NVMe storage reproduces the
~3x speedup (threaded vs multi-process workers) that backs the resume claim.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src"))

from torch.utils.data import DataLoader  # noqa: E402

from byu.data.threaded_loader import ThreadedDataLoader, yolo_collate  # noqa: E402
from byu.data.tomo_dataset import SyntheticTomoDataset, prepare_synthetic_tomograms  # noqa: E402


def _run_iter_and_time(loader, device: torch.device) -> float:
    """Walk the loader once, moving each batch to ``device``. Return elapsed seconds."""
    start = time.perf_counter()
    for imgs, _labels in loader:
        if device.type == "cuda":
            imgs = imgs.to(device, non_blocking=True)
            torch.cuda.synchronize()
    return time.perf_counter() - start


def _bench(loader_factory, n_passes: int, device: torch.device) -> list[float]:
    times: list[float] = []
    for _ in range(n_passes):
        loader = loader_factory()
        times.append(_run_iter_and_time(loader, device))
    return times


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=Path("./synthetic_tomos"))
    parser.add_argument("--n-tomos", type=int, default=3, help="Number of tomograms")
    parser.add_argument("--n-slices-per-tomo", type=int, default=64, help="Slices per tomo (synthetic only)")
    parser.add_argument("--slice-hw", type=int, nargs=2, default=(256, 256), metavar=("H", "W"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8, help="Workers for the DataLoader comparison")
    parser.add_argument("--num-threads", type=int, default=8, help="Threads for ThreadedDataLoader")
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--n-passes", type=int, default=3, help="Number of full epochs per config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=HERE.parent / "benchmarks")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {args.n_tomos} synthetic tomograms under {args.data_root} ...")
    tomo_ids = prepare_synthetic_tomograms(
        args.data_root,
        n_tomos=args.n_tomos,
        n_slices=args.n_slices_per_tomo,
        shape_hw=tuple(args.slice_hw),
    )
    z_indices = list(range(1, args.n_slices_per_tomo - 1))
    dataset = SyntheticTomoDataset(
        args.data_root, tomo_ids, z_indices, n_slices=3, shape_hw=tuple(args.slice_hw)
    )
    print(f"Dataset: {len(dataset)} samples, batch_size={args.batch_size}, device={device}")

    def dl_nw0():
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=yolo_collate)

    def dl_nwN():
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            persistent_workers=False,
            collate_fn=yolo_collate,
        )

    def threaded():
        return ThreadedDataLoader(
            dataset,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
            prefetch=args.prefetch,
        )

    configs = {
        "DataLoader(num_workers=0)": dl_nw0,
        f"DataLoader(num_workers={args.num_workers})": dl_nwN,
        f"ThreadedDataLoader(num_threads={args.num_threads})": threaded,
    }

    results: dict[str, dict[str, object]] = {}
    for name, factory in configs.items():
        print(f"\n[{name}]")
        times = _bench(factory, args.n_passes, device)
        mean, std = float(np.mean(times)), float(np.std(times))
        results[name] = {"times_s": times, "mean_s": mean, "std_s": std}
        print(f"   per-pass: {[f'{t:.2f}s' for t in times]}")
        print(f"   mean:     {mean:.2f}s +/- {std:.2f}s")

    baseline_name = f"DataLoader(num_workers={args.num_workers})"
    threaded_name = f"ThreadedDataLoader(num_threads={args.num_threads})"
    if baseline_name in results and threaded_name in results:
        base_mean = results[baseline_name]["mean_s"]
        thread_mean = results[threaded_name]["mean_s"]
        speedup = (base_mean - thread_mean) / base_mean * 100
        print(
            f"\nThreaded vs num_workers={args.num_workers}: {base_mean:.1f}s -> {thread_mean:.1f}s "
            f"({speedup:+.1f}%)"
        )

    _write_markdown(args.out_dir / "dataloader_results.md", args, results)
    _write_json(args.out_dir / "dataloader_results.json", args, results)
    _write_plot(args.out_dir / "dataloader_results.png", results)
    print(f"\nWrote results to {args.out_dir}/")
    return 0


def _write_markdown(path: Path, args, results: dict[str, dict]) -> None:
    lines = [
        "# DataLoader benchmark",
        "",
        f"- Device: `{args.device}`",
        f"- Tomograms: `{args.n_tomos}` x `{args.n_slices_per_tomo}` slices of "
        f"`{args.slice_hw[0]}x{args.slice_hw[1]}`",
        f"- Batch size: `{args.batch_size}`; passes per config: `{args.n_passes}`",
        "",
        "| Config | per-pass (s) | mean (s) | std (s) |",
        "|---|---|---|---|",
    ]
    for name, rec in results.items():
        times_fmt = ", ".join(f"{t:.2f}" for t in rec["times_s"])
        lines.append(f"| `{name}` | {times_fmt} | {rec['mean_s']:.2f} | {rec['std_s']:.2f} |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n")


def _write_json(path: Path, args, results) -> None:
    payload = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2))


def _write_plot(path: Path, results) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib unavailable, skipping {path}")
        return
    names = list(results.keys())
    means = [r["mean_s"] for r in results.values()]
    stds = [r["std_s"] for r in results.values()]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xpos = np.arange(len(names))
    ax.bar(xpos, means, yerr=stds, capsize=5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=10, ha="right")
    ax.set_ylabel("Wall time (s), lower is better")
    ax.set_title("DataLoader throughput comparison")
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.2f}s", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
