"""Query-centred, half-open, boundary-truncated conditional GQA.

K/V planes are supplied in semantic order by the caller (conditions, then
video history and current frame). Each plane has its own spatial window.
The CUDA path has fused Triton forward and backward kernels. The chunked
SDPA path is a portable numerical reference, not a high-resolution backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ._flashvsr_ops import _RotaryEmbedding3D

Window = tuple[int, int]


def _window(value: Sequence[int]) -> Window:
    if len(value) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 or v % 2 for v in value):
        raise ValueError(f"Spatial windows must contain two positive even integers: {value}.")
    return int(value[0]), int(value[1])


@dataclass(frozen=True)
class LocalAttentionConfig:
    video: Window = (24, 24)
    start: Window = (48, 48)
    end: Window = (48, 48)
    condition_self: Window = (48, 48)
    backend: str = "auto"

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> LocalAttentionConfig | None:
        if config is None:
            return None
        options = dict(config)
        for name, expected in (
            ("type", "query_centered"),
            ("interval", "half_open"),
            ("boundary", "truncated"),
        ):
            value = options.pop(name, expected)
            if value != expected:
                raise ValueError(f"spatial_attention.{name} must be {expected!r}, got {value!r}.")
        backend = options.pop("backend", "auto")
        if backend not in ("auto", "triton", "sdpa"):
            raise ValueError("Local attention backend must be auto, triton, or sdpa.")
        windows = {
            attr: _window(options.pop(name, default))
            for attr, name, default in (
                ("video", "video_window_size", (24, 24)),
                ("start", "start_condition_window_size", (48, 48)),
                ("end", "end_condition_window_size", (48, 48)),
                ("condition_self", "condition_self_window_size", (48, 48)),
            )
        }
        if options:
            raise ValueError(f"Unknown spatial_attention options: {sorted(options)}.")
        return cls(**windows, backend=backend)


def local_attention_mask(
    height: int,
    width: int,
    windows: tuple[Window, ...],
    *,
    device: torch.device,
    query_start: int = 0,
    query_end: int | None = None,
    query_frames: int = 1,
    condition_planes: int | None = None,
    temporal_window: int = 1,
) -> torch.Tensor:
    """Small/reference mask [queries, planes * H * W], True means visible."""
    n = height * width
    q = torch.arange(
        query_start, query_frames * n if query_end is None else query_end, device=device
    )
    qt = q // n
    q = q % n
    k = torch.arange(n, device=device)
    dy = k[None, :] // width - q[:, None] // width
    dx = k[None, :] % width - q[:, None] % width
    return torch.cat(
        [
            (dy >= -wh // 2)
            & (dy < wh // 2)
            & (dx >= -ww // 2)
            & (dx < ww // 2)
            & (
                torch.ones((q.numel(), 1), dtype=torch.bool, device=device)
                if condition_planes is None or plane < condition_planes
                else (
                    (plane - condition_planes <= qt[:, None])
                    & (plane - condition_planes > qt[:, None] - temporal_window)
                )
            )
            for plane, (wh, ww) in enumerate(windows)
        ],
        dim=-1,
    )


def local_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    height: int,
    width: int,
    windows: tuple[Window, ...],
    backend: str = "auto",
    query_frames: int = 1,
    condition_planes: int | None = None,
    temporal_window: int = 1,
    kv_plane_mask: torch.Tensor | None = None,
    relative_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Q=[B,Hq,T*H*W,D], K/V=[B,Hkv,F*H*W,D], with caller-applied spatial RoPE.

    condition_planes=None selects a single-query-frame operation over all
    supplied KV planes. Otherwise F=C+T and video keys obey the temporal window.
    kv_plane_mask is optional boolean [B,F]; False removes a complete KV plane
    from the softmax for that sample, including its K/V gradients.
    For parallel video, relative_rope=(cos,sin) has shape [temporal_window,Dt].
    Row lag rotates the first Dt key channels by -lag; conditions are untouched.
    Q/K must not already include temporal RoPE when relative_rope is supplied.
    The tables are fixed (non-trainable), with adjacent even/odd channel pairs.
    """
    n = height * width
    if query_frames < 1 or temporal_window < 1:
        raise ValueError("Positive query_frames and temporal_window are required.")
    if condition_planes is None:
        if query_frames != 1:
            raise ValueError("Multiple query frames require explicit condition_planes.")
    elif condition_planes < 0 or len(windows) != condition_planes + query_frames:
        raise ValueError("Parallel KV must contain condition planes followed by all video frames.")
    elif any(w != windows[condition_planes] for w in windows[condition_planes:]):
        raise ValueError("All parallel video planes must use the same spatial window.")
    if min(height, width) <= 0 or not windows:
        raise ValueError("Positive spatial dimensions and at least one KV plane are required.")
    for window in windows:
        _window(window)
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Local attention expects 4-D Q/K/V with matching K/V.")
    if (
        query.shape[0] != key.shape[0]
        or query.shape[2] != query_frames * n
        or key.shape[2] != len(windows) * n
        or query.shape[-1] != key.shape[-1]
        or query.shape[1] % key.shape[1]
    ):
        raise ValueError("Q/K/V do not match the spatial grid, planes, or GQA grouping.")
    if (
        query.device != key.device
        or query.device != value.device
        or query.dtype != key.dtype
        or query.dtype != value.dtype
    ):
        raise ValueError("Q/K/V must share a device and dtype.")
    if kv_plane_mask is not None and (
        kv_plane_mask.shape != (query.shape[0], len(windows))
        or kv_plane_mask.dtype != torch.bool
        or kv_plane_mask.device != query.device
    ):
        raise ValueError("kv_plane_mask must be boolean [B,planes] on the Q/K/V device.")
    if relative_rope is not None:
        if condition_planes is None or len(relative_rope) != 2:
            raise ValueError("Relative RoPE requires parallel condition_planes and two tables.")
        cosine, sine = relative_rope
        if (
            cosine.ndim != 2
            or cosine.shape != sine.shape
            or cosine.shape[0] != temporal_window
            or cosine.shape[1] < 2
            or cosine.shape[1] > query.shape[-1]
            or cosine.shape[1] % 2
            or any(
                t.device != query.device or t.dtype != query.dtype or t.requires_grad
                for t in relative_rope
            )
        ):
            raise ValueError(
                "Relative RoPE tables must be fixed [window,even_dim] on the Q/K dtype/device."
            )

    if backend not in ("auto", "triton", "sdpa"):
        raise ValueError(f"Unknown local attention backend: {backend}.")
    supported = (
        query.is_cuda and query.dtype in (torch.bfloat16, torch.float16) and query.shape[-1] == 64
    )
    if backend == "triton" or (backend == "auto" and supported):
        if not supported:
            raise ValueError("Triton local GQA requires CUDA BF16/FP16 and head_dim=64.")
        from ._local_attention_triton import triton_local_attention

        return triton_local_attention(
            query,
            key,
            value,
            height,
            width,
            windows,
            query_frames,
            condition_planes,
            temporal_window,
            kv_plane_mask,
            relative_rope,
        )
    if relative_rope is not None:
        return _relative_reference(
            query,
            key,
            value,
            height=height,
            width=width,
            windows=windows,
            query_frames=query_frames,
            condition_planes=condition_planes,
            temporal_window=temporal_window,
            kv_plane_mask=kv_plane_mask,
            relative_rope=relative_rope,
        )
    key_mask = (
        kv_plane_mask.repeat_interleave(n, dim=1)[:, None, None, :]
        if kv_plane_mask is not None
        else None
    )
    outputs = []
    for start in range(0, query_frames * n, 64):
        mask = local_attention_mask(
            height,
            width,
            windows,
            device=query.device,
            query_start=start,
            query_end=min(start + 64, query_frames * n),
            query_frames=query_frames,
            condition_planes=condition_planes,
            temporal_window=temporal_window,
        )
        if key_mask is not None:
            mask = mask[None, None, :, :] & key_mask
        outputs.append(
            F.scaled_dot_product_attention(
                query[:, :, start : start + 64],
                key,
                value,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        )
    return torch.cat(outputs, dim=2)


def _relative_reference(
    query,
    key,
    value,
    *,
    height,
    width,
    windows,
    query_frames,
    condition_planes,
    temporal_window,
    kv_plane_mask,
    relative_rope,
):
    """Small autograd oracle: rotate each frame's visible keys then run joint SDPA."""
    n = height * width
    cp = condition_planes
    temporal_dim = relative_rope[0].shape[-1]
    outputs = []
    for t in range(query_frames):
        first = max(0, t - temporal_window + 1)
        start, stop = cp + first, cp + t + 1
        lags = torch.arange(t - first, -1, -1, device=query.device)
        tables = tuple(table[lags].repeat_interleave(n, dim=0) for table in relative_rope)
        video_key = key[:, :, start * n : stop * n]
        rotated_key = torch.cat(
            (
                _RotaryEmbedding3D.apply(video_key[..., :temporal_dim], *tables),
                video_key[..., temporal_dim:],
            ),
            dim=-1,
        )
        selected_key = torch.cat((key[:, :, : cp * n], rotated_key), dim=2)
        selected_value = torch.cat(
            (value[:, :, : cp * n], value[:, :, start * n : stop * n]), dim=2
        )
        selected_mask = (
            torch.cat((kv_plane_mask[:, :cp], kv_plane_mask[:, start:stop]), dim=1)
            if kv_plane_mask is not None
            else None
        )
        outputs.append(
            local_attention(
                query[:, :, t * n : (t + 1) * n],
                selected_key,
                selected_value,
                height=height,
                width=width,
                windows=windows[:cp] + windows[start:stop],
                backend="sdpa",
                kv_plane_mask=selected_mask,
            )
        )
    return torch.cat(outputs, dim=2)
