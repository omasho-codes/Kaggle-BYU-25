"""
Weight-tied multi-scale filter clones.

Both variants in this module share **one** ``nn.Conv2d`` weight tensor across multiple
effective scales, so gradients from every scale accumulate into the same parameter tensor
during backprop. This is the "weight-tied ... aggregate multi-scale gradients" idea that
sits on the terminal layers of the MHAF-YOLO head in the BYU-25 solution.

Two variants are provided:

- :class:`DilatedCloneConv` - one :class:`nn.Conv2d` applied repeatedly at different
  dilation rates. Effective receptive fields are ``k + (k-1)*(d-1)`` per branch.
- :class:`GaussianPyramidConv` - one :class:`nn.Conv2d` applied to the feature map at
  multiple spatial resolutions (produced by :func:`F.avg_pool2d`), with each branch's
  output upsampled back to the original resolution.

Both expose the same ``(c_in, c_out, k, ...)`` signature and a ``forward(x)`` returning a
tensor of shape ``(B, c_out, H, W)``.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

_AggMode = Literal["sum", "mean", "concat"]


def _aggregate(parts: list[torch.Tensor], mode: _AggMode, proj: nn.Module | None = None) -> torch.Tensor:
    if mode == "sum":
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out
    if mode == "mean":
        stacked = torch.stack(parts, dim=0)
        return stacked.mean(dim=0)
    if mode == "concat":
        assert proj is not None, "concat aggregation requires a projection"
        return proj(torch.cat(parts, dim=1))
    raise ValueError(f"Unknown aggregation mode: {mode}")


class DilatedCloneConv(nn.Module):
    """Weight-tied multi-scale conv: same KxK kernel at multiple dilations.

    Args:
        c_in: input channel count (must equal ``c_out`` for cheap residual wiring).
        c_out: output channel count.
        k: kernel size (odd int, default 3).
        dilations: tuple of positive ints, one per clone. ``(1, 2, 3)`` by default.
        agg: aggregation mode across clones: ``"sum"``, ``"mean"``, or ``"concat"``.
        bias: whether to include a bias on the shared kernel.
        bn: whether to apply a shared :class:`nn.BatchNorm2d` after aggregation.
        act: activation applied after bn (``None`` for identity).
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        dilations: tuple[int, ...] = (1, 2, 3),
        agg: _AggMode = "sum",
        bias: bool = False,
        bn: bool = True,
        act: nn.Module | None = None,
    ):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("kernel size must be odd")
        if any(d < 1 for d in dilations):
            raise ValueError("dilations must be >= 1")
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.dilations = tuple(dilations)
        self.agg = agg
        self.shared = nn.Conv2d(c_in, c_out, kernel_size=k, bias=bias)
        self.proj: nn.Module | None = None
        if agg == "concat":
            self.proj = nn.Conv2d(c_out * len(dilations), c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out) if bn else nn.Identity()
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self.shared.weight, self.shared.bias
        outs: list[torch.Tensor] = []
        for d in self.dilations:
            padding = (self.k + (self.k - 1) * (d - 1)) // 2
            outs.append(F.conv2d(x, weight, bias, stride=1, padding=padding, dilation=d))
        y = _aggregate(outs, self.agg, self.proj)
        return self.act(self.bn(y))

    def extra_repr(self) -> str:
        return (
            f"c_in={self.c_in}, c_out={self.c_out}, k={self.k}, "
            f"dilations={self.dilations}, agg={self.agg}"
        )


class GaussianPyramidConv(nn.Module):
    """Weight-tied multi-scale conv: same KxK kernel applied at multiple pooling levels.

    Level ``i`` applies the shared conv to ``AvgPool2d(2**i)(x)``, then bilinearly upsamples
    back to the original spatial resolution before aggregation. Level 0 is the identity
    resolution.

    Args:
        c_in: input channel count.
        c_out: output channel count.
        k: kernel size (odd int, default 3).
        levels: number of pyramid levels (>= 1). Default 3 -> scales ``{1, 1/2, 1/4}``.
        agg: aggregation mode.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        levels: int = 3,
        agg: _AggMode = "sum",
        bias: bool = False,
        bn: bool = True,
        act: nn.Module | None = None,
    ):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("kernel size must be odd")
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.c_in = c_in
        self.c_out = c_out
        self.k = k
        self.levels = levels
        self.agg = agg
        self.shared = nn.Conv2d(c_in, c_out, kernel_size=k, padding=k // 2, bias=bias)
        self.proj: nn.Module | None = None
        if agg == "concat":
            self.proj = nn.Conv2d(c_out * levels, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out) if bn else nn.Identity()
        self.act = act if act is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        outs: list[torch.Tensor] = []
        for i in range(self.levels):
            if i == 0:
                xi = x
            else:
                stride = 2**i
                xi = F.avg_pool2d(x, kernel_size=stride, stride=stride, ceil_mode=True)
            yi = self.shared(xi)
            if yi.shape[-2:] != (h, w):
                yi = F.interpolate(yi, size=(h, w), mode="bilinear", align_corners=False)
            outs.append(yi)
        y = _aggregate(outs, self.agg, self.proj)
        return self.act(self.bn(y))

    def extra_repr(self) -> str:
        return f"c_in={self.c_in}, c_out={self.c_out}, k={self.k}, " f"levels={self.levels}, agg={self.agg}"


def build_wt_module(
    variant: str,
    c_in: int,
    c_out: int,
    k: int = 3,
    **kwargs,
) -> nn.Module:
    """Factory that maps a variant name to one of :class:`DilatedCloneConv` /
    :class:`GaussianPyramidConv`.

    ``variant`` is case-insensitive. ``"baseline"`` returns an identity conv (no WT module)
    so that YAMLs can select the ablation branch without branching at load time.
    """
    v = variant.lower()
    if v in ("dilated", "wt_dilated"):
        dilations = kwargs.pop("dilations", (1, 2, 3))
        return DilatedCloneConv(c_in, c_out, k=k, dilations=tuple(dilations), **kwargs)
    if v in ("gpyramid", "gaussian", "wt_gpyramid"):
        levels = kwargs.pop("levels", 3)
        return GaussianPyramidConv(c_in, c_out, k=k, levels=levels, **kwargs)
    if v in ("baseline", "none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown WT variant: {variant!r}")


__all__ = ["DilatedCloneConv", "GaussianPyramidConv", "build_wt_module"]
