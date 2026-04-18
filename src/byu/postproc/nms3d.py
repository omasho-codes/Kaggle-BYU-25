"""
3D NMS + 2D-to-3D linking for the BYU flagellar-motor pipeline.

This module distils the 2D-slice-to-3D-object linking + NMS logic used in the original
Kaggle submission. The original had the LB score jump from 0.754
to 0.812 once 3D NMS replaced per-slice 2D NMS + nearest-neighbor merging.

Core idea:

1. For each tomogram, inference on its slices yields per-slice 2D boxes in xywhc format.
2. Sweep z in increasing order. For each slice, try to attach each 2D box to an existing
   3D track via 2D IoU on the track's last box, breaking ties by max IoU.
3. A track whose ``last_seen_z`` lags by more than ``max_missed_slices`` is closed.
4. Tracks that survived fewer than ``min_slices_for_3d`` slices are discarded; otherwise
   they produce a 3D box (axis-aligned over all seen slices) with a confidence that is
   boosted by length.
5. Run class-aware greedy 3D NMS with ``nms_iou_threshold_3d``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

Box3D = tuple[float, float, float, float, float, float, int, float]  # x1,y1,z1,x2,y2,z2,cls,conf


def xywh_to_xyxy(xc: float, yc: float, w: float, h: float) -> tuple[float, float, float, float]:
    return xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2


def calculate_2d_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    xb = min(a[2], b[2])
    yb = min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def calculate_3d_iou(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
) -> float:
    xa = max(a[0], b[0])
    ya = max(a[1], b[1])
    za = max(a[2], b[2])
    xb = min(a[3], b[3])
    yb = min(a[4], b[4])
    zb = min(a[5], b[5])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya) * max(0.0, zb - za)
    if inter == 0.0:
        return 0.0
    vol_a = (a[3] - a[0]) * (a[4] - a[1]) * (a[5] - a[2])
    vol_b = (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2])
    return inter / (vol_a + vol_b - inter)


@dataclass
class Track3D:
    """A growing 3D track assembled from per-slice 2D detections."""

    track_id: int
    class_id: int
    boxes_2d_by_z: dict[int, tuple[float, float, float, float, float]] = field(default_factory=dict)
    last_seen_z: int = -1
    confidences: list[float] = field(default_factory=list)

    @classmethod
    def _counter(cls) -> int:
        cls.__last_id = getattr(cls, "_Track3D__last_id", -1) + 1
        return cls.__last_id

    @classmethod
    def reset(cls) -> None:
        cls.__last_id = -1

    @classmethod
    def create(cls, det: tuple[float, float, float, float, float], z: int, class_id: int) -> Track3D:
        t = cls(track_id=cls._counter(), class_id=class_id)
        t._add_unchecked(det, z)
        return t

    def _add_unchecked(self, det: tuple[float, float, float, float, float], z: int) -> None:
        x1, y1, x2, y2 = xywh_to_xyxy(*det[:4])
        self.boxes_2d_by_z[z] = (x1, y1, x2, y2, det[4])
        self.last_seen_z = z
        self.confidences.append(det[4])

    def add(self, det: tuple[float, float, float, float, float], z: int) -> None:
        self._add_unchecked(det, z)

    def last_box_xyxy(self) -> tuple[float, float, float, float] | None:
        if not self.boxes_2d_by_z:
            return None
        return self.boxes_2d_by_z[self.last_seen_z][:4]

    def finalize(
        self,
        min_slices_for_3d: int,
        confidence_boost_factor: float,
    ) -> Box3D | None:
        n = len(self.boxes_2d_by_z)
        if n < min_slices_for_3d:
            return None
        boxes = list(self.boxes_2d_by_z.values())
        x1 = float(np.mean([b[0] for b in boxes]))
        y1 = float(np.mean([b[1] for b in boxes]))
        x2 = float(np.mean([b[2] for b in boxes]))
        y2 = float(np.mean([b[3] for b in boxes]))
        zs = sorted(self.boxes_2d_by_z.keys())
        z1 = float(zs[0])
        z2 = float(zs[-1])
        avg = float(np.mean(self.confidences))
        boost = confidence_boost_factor * max(0, n - min_slices_for_3d)
        conf = min(1.0, avg + boost)
        return (x1, y1, z1, x2, y2, z2, self.class_id, conf)


def nms3d(boxes: Iterable[Box3D], iou_threshold: float = 0.2) -> list[Box3D]:
    """Class-aware greedy 3D NMS on a list of ``Box3D`` tuples."""
    pending = sorted(list(boxes), key=lambda b: b[7], reverse=True)
    kept: list[Box3D] = []
    while pending:
        current = pending.pop(0)
        kept.append(current)
        pending = [
            b for b in pending if b[6] != current[6] or calculate_3d_iou(current[:6], b[:6]) < iou_threshold
        ]
    return kept


def merge_2d_predictions_to_3d(
    per_slice_xywhc: dict[int, list[tuple[float, float, float, float, int, float]]],
    iou_threshold_2d_link: float = 0.3,
    max_missed_slices: int = 1,
    min_slices_for_3d: int = 3,
    confidence_boost_factor: float = 0.2,
    nms_iou_threshold_3d: float = 0.2,
    track_confidence_threshold: float = 0.0,
) -> list[Box3D]:
    """Link per-slice 2D boxes across z, finalise 3D tracks, apply 3D NMS.

    Args:
        per_slice_xywhc: mapping from ``z`` to a list of ``(xc, yc, w, h, cls, conf)``.
        iou_threshold_2d_link: minimum 2D IoU to attach a new detection to an active
            track.
        max_missed_slices: how many z-steps a track may survive without a match before
            it is closed.
        min_slices_for_3d: minimum number of slices a track must span to become a 3D box.
        confidence_boost_factor: added confidence per extra slice beyond
            ``min_slices_for_3d``.
        nms_iou_threshold_3d: 3D IoU threshold for the final NMS.
        track_confidence_threshold: drop finalised tracks below this confidence.
    """
    Track3D.reset()
    active: list[Track3D] = []
    completed: list[Track3D] = []

    for z in sorted(per_slice_xywhc.keys()):
        dets = per_slice_xywhc.get(z, [])
        processed = [
            (xywh_to_xyxy(xc, yc, w, h), int(cls), float(conf), (xc, yc, w, h, conf))
            for xc, yc, w, h, cls, conf in dets
        ]
        matched = [False] * len(processed)

        for i in range(len(active) - 1, -1, -1):
            track = active[i]
            if z - track.last_seen_z > max_missed_slices:
                completed.append(active.pop(i))
                continue
            last = track.last_box_xyxy()
            if last is None:
                continue
            best_idx, best_iou = -1, 0.0
            for j, (xyxy, cls, _, _) in enumerate(processed):
                if matched[j] or cls != track.class_id:
                    continue
                iou = calculate_2d_iou(last, xyxy)
                if iou > iou_threshold_2d_link and iou > best_iou:
                    best_idx, best_iou = j, iou
            if best_idx >= 0:
                track.add(processed[best_idx][3], z)
                matched[best_idx] = True

        for j, (_, cls, _, det) in enumerate(processed):
            if not matched[j]:
                active.append(Track3D.create(det, z, cls))

    completed.extend(active)

    pre_nms: list[Box3D] = []
    for track in completed:
        box = track.finalize(min_slices_for_3d, confidence_boost_factor)
        if box is not None and box[7] >= track_confidence_threshold:
            pre_nms.append(box)
    return nms3d(pre_nms, iou_threshold=nms_iou_threshold_3d)


__all__ = [
    "Box3D",
    "Track3D",
    "xywh_to_xyxy",
    "calculate_2d_iou",
    "calculate_3d_iou",
    "nms3d",
    "merge_2d_predictions_to_3d",
]
