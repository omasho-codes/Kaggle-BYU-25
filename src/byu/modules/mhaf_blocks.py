"""
Vendored MHAF-YOLO-specific blocks.

Source: https://github.com/omasho-codes/MHAF-YOLO (fork of the upstream MHAF-YOLO paper repo),
which in turn extends Ultralytics with ``RepHMS``, ``PSA``, ``SCDown``, ``AVG`` and
``UniRepLKNetBlock``. We vendor only these classes (not the whole framework) so the BYU repo
can install ``ultralytics`` from PyPI and plug these blocks in via :func:`register_mhaf_modules`.

The implementations here are minimally adapted from the upstream fork:
- use the upstream ``Conv`` from ``ultralytics.nn.modules.conv`` instead of the fork's local copy
- drop training-mode-only dead code paths that were unused in BYU experiments
"""

from __future__ import annotations

import collections.abc
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_to_2tuple = _ntuple(2)


def _get_bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels)


def _get_conv2d(in_c: int, out_c: int, k, stride, padding, dilation, groups, bias: bool) -> nn.Conv2d:
    k = _to_2tuple(k)
    padding = (k[0] // 2, k[1] // 2) if padding is None else _to_2tuple(padding)
    return nn.Conv2d(
        in_channels=in_c,
        out_channels=out_c,
        kernel_size=k,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel = conv.weight
    gamma, beta, eps = bn.weight, bn.bias, bn.eps
    std = (bn.running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)
    return kernel * t, beta - bn.running_mean * gamma / std


def _convert_dilated_to_nondilated(kernel: torch.Tensor, dilate_rate: int) -> torch.Tensor:
    identity = torch.ones((1, 1, 1, 1), dtype=kernel.dtype, device=kernel.device)
    if kernel.size(1) == 1:
        return F.conv_transpose2d(kernel, identity, stride=dilate_rate)
    slices = [
        F.conv_transpose2d(kernel[:, i : i + 1, :, :], identity, stride=dilate_rate)
        for i in range(kernel.size(1))
    ]
    return torch.cat(slices, dim=1)


def _merge_dilated_into_large_kernel(large: torch.Tensor, dilated: torch.Tensor, rate: int) -> torch.Tensor:
    large_k = large.size(2)
    dilated_k = dilated.size(2)
    eq_size = rate * (dilated_k - 1) + 1
    equivalent = _convert_dilated_to_nondilated(dilated, rate)
    pad = large_k // 2 - eq_size // 2
    return large + F.pad(equivalent, [pad] * 4)


class AVG(nn.Module):
    """Adaptive average pool that downsamples by an integer factor."""

    def __init__(self, down_n: int = 2):
        super().__init__()
        self.down_n = down_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        out_size = np.array([int(h / self.down_n), int(w / self.down_n)])
        return F.adaptive_avg_pool2d(x, out_size.tolist())


class DilatedReparamBlock(nn.Module):
    """Dilated re-parametrisation block from UniRepLKNet.

    At training time it is a sum of one large-kernel depthwise conv plus several dilated
    branches, each with its own BN. The branches can be fused back into a single large
    depthwise conv at deploy time via :meth:`merge_dilated_branches`.
    """

    _KS_TO_BRANCHES: dict[int, tuple[list[int], list[int]]] = {
        17: ([5, 9, 3, 3, 3], [1, 2, 4, 5, 7]),
        15: ([5, 7, 3, 3, 3], [1, 2, 3, 5, 7]),
        13: ([5, 7, 3, 3, 3], [1, 2, 3, 4, 5]),
        11: ([5, 5, 3, 3, 3], [1, 2, 3, 4, 5]),
        9: ([7, 5, 3], [1, 1, 1]),
        7: ([5, 3], [1, 1]),
        5: ([3, 1], [1, 1]),
        3: ([3, 1], [1, 1]),
    }

    def __init__(self, channels: int, kernel_size: int, deploy: bool = False):
        super().__init__()
        self.lk_origin = _get_conv2d(
            channels, channels, kernel_size, 1, kernel_size // 2, 1, groups=channels, bias=deploy
        )
        if kernel_size not in self._KS_TO_BRANCHES:
            raise ValueError(f"DilatedReparamBlock does not support kernel_size={kernel_size}")
        self.kernel_sizes, self.dilates = self._KS_TO_BRANCHES[kernel_size]
        if not deploy:
            self.origin_bn = _get_bn(channels)
            for k, r in zip(self.kernel_sizes, self.dilates, strict=False):
                self.__setattr__(
                    f"dil_conv_k{k}_{r}",
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=k,
                        stride=1,
                        padding=(r * (k - 1) + 1) // 2,
                        dilation=r,
                        groups=channels,
                        bias=False,
                    ),
                )
                self.__setattr__(f"dil_bn_k{k}_{r}", _get_bn(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "origin_bn"):
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates, strict=False):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self) -> None:
        if not hasattr(self, "origin_bn"):
            return
        origin_k, origin_b = _fuse_bn(self.lk_origin, self.origin_bn)
        for k, r in zip(self.kernel_sizes, self.dilates, strict=False):
            conv = getattr(self, f"dil_conv_k{k}_{r}")
            bn = getattr(self, f"dil_bn_k{k}_{r}")
            branch_k, branch_b = _fuse_bn(conv, bn)
            origin_k = _merge_dilated_into_large_kernel(origin_k, branch_k, r)
            origin_b = origin_b + branch_b
        merged = _get_conv2d(
            origin_k.size(0),
            origin_k.size(0),
            origin_k.size(2),
            1,
            origin_k.size(2) // 2,
            1,
            origin_k.size(0),
            True,
        )
        merged.weight.data = origin_k
        merged.bias.data = origin_b
        self.lk_origin = merged
        del self.origin_bn
        for k, r in zip(self.kernel_sizes, self.dilates, strict=False):
            delattr(self, f"dil_conv_k{k}_{r}")
            delattr(self, f"dil_bn_k{k}_{r}")


class UniRepLKNetBlock(nn.Module):
    """Depthwise reparametrised large-kernel block used in MHAF-YOLO terminal clones."""

    def __init__(self, dim: int, kernel_size: int, deploy: bool = False):
        super().__init__()
        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 3:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy)
        else:
            raise ValueError(f"Unsupported kernel_size={kernel_size}")
        self.norm = nn.Identity() if deploy or kernel_size == 0 else _get_bn(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.dwconv(x))


class _DepthBottleneckUniv2(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int,
        shortcut: bool = True,
        kersize: int = 5,
        expansion_depth: int = 1,
        use_depthwise: bool = True,
    ):
        super().__init__()
        mid = int(in_c * expansion_depth)
        self.conv1 = Conv(in_c, mid, k=1)
        if use_depthwise:
            self.conv2 = UniRepLKNetBlock(mid, kernel_size=kersize)
            self.act = nn.SiLU()
            self.one_conv = Conv(mid, mid, k=1)
            self.conv3 = UniRepLKNetBlock(mid, kernel_size=kersize)
            self.act1 = nn.SiLU()
            self.one_conv2 = Conv(mid, out_c, k=1)
        else:
            self.conv2 = Conv(out_c, out_c, k=3)
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act(self.conv2(y))
        y = self.one_conv(y)
        y = self.act1(self.conv3(y))
        return self.one_conv2(y)


class RepHMS(nn.Module):
    """Multi-Branch Heterogeneous Auxiliary Fusion block from MHAF-YOLO."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        width: int = 3,
        depth: int = 1,
        depth_expansion: int = 2,
        kersize: int = 5,
        shortcut: bool = True,
        expansion: float = 0.5,
        use_depthwise: bool = True,
    ):
        super().__init__()
        self.width = width
        self.depth = depth
        c1 = int(out_c * expansion) * width
        self.c_ = int(out_c * expansion)
        self.conv1 = Conv(in_c, c1, k=1)
        self.rep_elan = nn.ModuleList()
        for _ in range(width - 1):
            self.rep_elan.append(
                nn.ModuleList(
                    _DepthBottleneckUniv2(self.c_, self.c_, shortcut, kersize, depth_expansion, use_depthwise)
                    for _ in range(depth)
                )
            )
        self.conv2 = Conv(self.c_ * 1 + self.c_ * (width - 1) * depth, out_c, k=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x_out = [x[:, i * self.c_ : (i + 1) * self.c_] for i in range(self.width)]
        x_out[1] = x_out[1] + x_out[0]
        cascade: list[torch.Tensor] = []
        elan = [x_out[0]]
        for i in range(self.width - 1):
            for j in range(self.depth):
                if i > 0:
                    x_out[i + 1] = x_out[i + 1] + cascade[j]
                    if j == self.depth - 1:
                        cascade = [cascade[-1]] if self.depth > 1 else []
                x_out[i + 1] = self.rep_elan[i][j](x_out[i + 1])
                elan.append(x_out[i + 1])
                if i < self.width - 2:
                    cascade.append(x_out[i + 1])
        return self.conv2(torch.cat(elan, 1))


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        h = dim + self.key_dim * num_heads * 2
        self.qkv = Conv(dim, h, k=1, act=False)
        self.proj = Conv(dim, dim, k=1, act=False)
        self.pe = Conv(dim, dim, k=3, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x)
        q, k, v = qkv.view(b, self.num_heads, self.key_dim * 2 + self.head_dim, n).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class PSA(nn.Module):
    """Partial Self-Attention block used in MHAF-YOLO backbone tail."""

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2, "PSA expects c1 == c2"
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, k=1)
        self.cv2 = Conv(2 * self.c, c1, k=1)
        self.attn = _Attention(self.c, attn_ratio=0.5, num_heads=max(1, self.c // 64))
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, k=1), Conv(self.c * 2, self.c, k=1, act=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class SCDown(nn.Module):
    """Spatial-Channel decoupled downsample block."""

    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = Conv(c1, c2, k=1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


__all__ = ["AVG", "UniRepLKNetBlock", "DilatedReparamBlock", "RepHMS", "PSA", "SCDown"]
