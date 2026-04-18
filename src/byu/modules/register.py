"""
Register BYU / MHAF custom modules with Ultralytics' YAML model parser.

Ultralytics' ``parse_model`` in :mod:`ultralytics.nn.tasks` resolves class names by
``globals()[m]`` and then dispatches on class identity via a series of
``if m in frozenset({...})`` checks. The frozensets are local to the function, so to make
new modules work we (a) inject the class symbols into the ``tasks`` module namespace for
name resolution, and (b) monkey-patch ``parse_model`` with a drop-in reimplementation that
recognises the new classes.

The reimplementation mirrors Ultralytics 8.4.x parse_model exactly for upstream modules
(so YAMLs that don't use our custom blocks continue to work), and adds three new paths:

- :class:`RepHMS` -> Conv-like dispatch (``args = [c1, c2, *args[1:]]``, no repeat arg)
- :class:`AVG` -> pass-through dispatch (``c2 = ch[f]``)
- :class:`WTMultiScaleDetect` -> ``v10Detect``-like dispatch

Everything here is idempotent - :func:`register_all` is safe to call multiple times.
"""

from __future__ import annotations

import ast
import contextlib

import torch
import torch.nn as nn

from .mhaf_blocks import AVG, PSA, RepHMS, SCDown, UniRepLKNetBlock
from .wt_detect_head import WTMultiScaleDetect
from .wt_ms_conv import DilatedCloneConv, GaussianPyramidConv

_SENTINEL = "_byu_registered"


def register_mhaf_modules() -> None:
    """Inject vendored MHAF blocks into :mod:`ultralytics.nn.tasks` globals."""
    import ultralytics.nn.tasks as tasks

    tasks.RepHMS = RepHMS
    tasks.AVG = AVG
    tasks.UniRepLKNetBlock = UniRepLKNetBlock
    if not hasattr(tasks, "PSA"):
        tasks.PSA = PSA
    if not hasattr(tasks, "SCDown"):
        tasks.SCDown = SCDown


def register_wt_modules() -> None:
    """Inject weight-tied modules + detect head into :mod:`ultralytics.nn.tasks` globals."""
    import ultralytics.nn.tasks as tasks

    tasks.WTMultiScaleDetect = WTMultiScaleDetect
    tasks.DilatedCloneConv = DilatedCloneConv
    tasks.GaussianPyramidConv = GaussianPyramidConv


def _build_parse_model():
    """Build the replacement ``parse_model`` closure, lazily importing symbols from
    Ultralytics so we see whatever version is installed."""
    from ultralytics.utils import LOGGER, colorstr

    try:
        from ultralytics.utils.torch_utils import make_divisible
    except ImportError:
        from ultralytics.nn.tasks import make_divisible
    from ultralytics.nn.modules import (
        AIFI,
        C1,
        C2,
        C3,
        C3TR,
        OBB,
        SPP,
        SPPELAN,
        SPPF,
        Bottleneck,
        BottleneckCSP,
        C2f,
        C2fAttn,
        C2fCIB,
        C3Ghost,
        C3x,
        CBFuse,
        CBLinear,
        Classify,
        Concat,
        Conv,
        ConvTranspose,
        Detect,
        DWConv,
        DWConvTranspose2d,
        Focus,
        GhostBottleneck,
        GhostConv,
        HGBlock,
        HGStem,
        ImagePoolingAttn,
        Pose,
        RepC3,
        RepNCSPELAN4,
        ResNetLayer,
        RTDETRDecoder,
        Segment,
        WorldDetect,
    )

    try:
        from ultralytics.nn.modules import ADown
    except ImportError:
        ADown = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules import AConv
    except ImportError:
        AConv = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules import A2C2f
    except ImportError:
        A2C2f = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules import C2PSA
    except ImportError:
        C2PSA = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules import C2fPSA
    except ImportError:
        C2fPSA = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules import C3k2
    except ImportError:
        C3k2 = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules import ELAN1
    except ImportError:
        ELAN1 = type("_Absent", (), {})
    try:
        from ultralytics.nn.modules.block import PSA as UltraPSA
    except ImportError:
        UltraPSA = PSA
    try:
        from ultralytics.nn.modules.block import SCDown as UltraSCDown
    except ImportError:
        UltraSCDown = SCDown
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules.head import v10Detect

    base_modules = frozenset(
        {
            Classify,
            Conv,
            ConvTranspose,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            C2fPSA,
            C2PSA,
            DWConv,
            Focus,
            BottleneckCSP,
            C1,
            C2,
            C2f,
            C3k2,
            RepNCSPELAN4,
            ELAN1,
            ADown,
            AConv,
            SPPELAN,
            C2fAttn,
            C3,
            C3TR,
            C3Ghost,
            torch.nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            RepC3,
            UltraPSA,
            UltraSCDown,
            C2fCIB,
            A2C2f,
            RepHMS,
        }
    )
    repeat_modules = frozenset(
        {BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3, C2fCIB, A2C2f, C3k2}
    )
    detect_modules = frozenset({Detect, WorldDetect, Segment, Pose, OBB, ImagePoolingAttn})

    def parse_model(d, ch, verbose=True):  # noqa: C901 (mirrors upstream density)
        max_channels = float("inf")
        nc = d.get("nc")
        act = d.get("activation")
        scales = d.get("scales")
        depth = d.get("depth_multiple", 1.0)
        width = d.get("width_multiple", 1.0)
        scale = d.get("scale")
        if scales:
            if not scale:
                scale = next(iter(scales.keys()))
                LOGGER.warning(f"no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]
        if act:
            Conv.default_act = eval(act)  # noqa: S307
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")
        if verbose:
            LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")

        ch_list = [ch]
        layers: list[nn.Module] = []
        save: list[int] = []
        c2 = ch_list[-1]

        for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
            if isinstance(m, str) and m.startswith("nn."):
                m_cls = getattr(torch.nn, m[3:])
            elif isinstance(m, str):
                if m not in tasks.__dict__:
                    raise KeyError(f"Unknown module name in YAML: {m!r}")
                m_cls = tasks.__dict__[m]
            else:
                m_cls = m

            local_lookup = {"nc": nc, "scale": scale, "width": width, "depth": depth}
            for j, a in enumerate(args):
                if isinstance(a, str):
                    if a in local_lookup:
                        args[j] = local_lookup[a]
                    else:
                        with contextlib.suppress(ValueError):
                            args[j] = ast.literal_eval(a)

            n = n_ = max(round(n * depth), 1) if n > 1 else n

            if m_cls in base_modules:
                c1, c2 = ch_list[f], args[0]
                if c2 != nc:
                    c2 = make_divisible(min(c2, max_channels) * width, 8)
                args = [c1, c2, *args[1:]]
                if m_cls in repeat_modules:
                    args.insert(2, n)
                    n = 1
            elif m_cls is AIFI:
                args = [ch_list[f], *args]
            elif m_cls in {HGStem, HGBlock}:
                c1, cm, c2 = ch_list[f], args[0], args[1]
                args = [c1, cm, c2, *args[2:]]
                if m_cls is HGBlock:
                    args.insert(4, n)
                    n = 1
            elif m_cls is ResNetLayer:
                c2 = args[1] if args[3] else args[1] * 4
            elif m_cls is nn.BatchNorm2d:
                args = [ch_list[f]]
            elif m_cls is AVG:
                c2 = ch_list[f]
            elif m_cls is Concat:
                c2 = sum(ch_list[x] for x in f)
            elif m_cls in detect_modules:
                args.append([ch_list[x] for x in f])
                if m_cls is Segment:
                    args[2] = make_divisible(min(args[2], max_channels) * width, 8)
            elif m_cls is v10Detect or m_cls is WTMultiScaleDetect:
                args.append([ch_list[x] for x in f])
            elif m_cls is RTDETRDecoder:
                args.insert(1, [ch_list[x] for x in f])
            elif m_cls is CBLinear:
                c2 = args[0]
                c1 = ch_list[f]
                args = [c1, c2, *args[1:]]
            elif m_cls is CBFuse:
                c2 = ch_list[f[-1]]
            else:
                c2 = ch_list[f]

            m_ = nn.Sequential(*(m_cls(*args) for _ in range(n))) if n > 1 else m_cls(*args)
            type_str = str(m_cls)[8:-2].replace("__main__.", "")
            m_cls.np = sum(p.numel() for p in m_.parameters())
            m_.i, m_.f, m_.type = i, f, type_str
            if verbose:
                LOGGER.info(f"{i:>3}{str(f):>20}{n_:>3}{m_cls.np:10.0f}  {type_str:<45}{str(args):<30}")
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch_list = []
            ch_list.append(c2)
        return nn.Sequential(*layers), sorted(save)

    return parse_model


def register_parse_model() -> None:
    """Swap Ultralytics' ``parse_model`` for our extended version."""
    import ultralytics.nn.tasks as tasks

    if getattr(tasks, _SENTINEL, False):
        return
    tasks._byu_original_parse_model = tasks.parse_model
    tasks.parse_model = _build_parse_model()
    setattr(tasks, _SENTINEL, True)


def register_all() -> None:
    """One-shot registration: MHAF blocks + WT modules + patched ``parse_model``."""
    register_mhaf_modules()
    register_wt_modules()
    register_parse_model()


__all__ = [
    "register_mhaf_modules",
    "register_wt_modules",
    "register_parse_model",
    "register_all",
]
