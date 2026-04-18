"""Detection head with weight-tied multi-scale filter clones on each feature map.

The parent class is :class:`ultralytics.nn.modules.head.v10Detect`, which itself extends
``Detect`` with a separate ``one2one`` branch. We keep everything ``v10Detect`` does and
simply pre-process each incoming feature map ``x[i]`` with a per-scale ``WT`` block. Since
the WT block returns ``(B, C, H, W)`` of the same shape as its input, no downstream
plumbing needs to change.

Motivation: the resume claim is "weight-tied, multi-scale filter clones in the terminal
layers to aggregate multi-scale gradients, guided by Grad-CAM analysis". Placing the WT
block *before* the ``v10Detect`` ``cv2``/``cv3`` 1x1 heads ensures the shared kernel gets
gradient contributions from all three pyramid levels (P3, P4, P5) as well as all scales
within each level.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.head import v10Detect

from .wt_ms_conv import build_wt_module


class WTMultiScaleDetect(v10Detect):
    """Drop-in ``v10Detect`` replacement that prepends a weight-tied multi-scale block
    to each incoming feature map.

    Args:
        nc: number of classes (same as ``v10Detect``).
        ch: per-scale channel count tuple (same as ``v10Detect``).
        variant: which WT module to use - ``"dilated"`` or ``"gpyramid"``.
        wt_k: kernel size for the shared conv (default 3).
        wt_dilations: dilation tuple for the dilated variant (default ``(1, 2, 3)``).
        wt_levels: pyramid depth for the gpyramid variant (default 3).
        wt_agg: aggregation mode across scales (default ``"sum"``).
    """

    def __init__(
        self,
        nc: int = 80,
        variant: str = "dilated",
        wt_k: int = 3,
        wt_dilations: tuple[int, ...] = (1, 2, 3),
        wt_levels: int = 3,
        wt_agg: str = "sum",
        ch: tuple[int, ...] = (),
    ):
        super().__init__(nc=nc, ch=ch)
        self.variant = variant
        kwargs = {"agg": wt_agg, "bn": True, "act": nn.SiLU()}
        if variant == "dilated":
            kwargs["dilations"] = wt_dilations
        elif variant == "gpyramid":
            kwargs["levels"] = wt_levels
        self.wt = nn.ModuleList([build_wt_module(variant, c, c, k=wt_k, **kwargs) for c in ch])

    def forward(self, x: list[torch.Tensor]):
        x = [wt(xi) for wt, xi in zip(self.wt, x, strict=True)]
        return super().forward(x)


__all__ = ["WTMultiScaleDetect"]
