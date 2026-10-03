"""Private Transformer operations used by conditional FlashVSR."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class _KVCache:
    """Unrotated, normalized keys and values for one VSR layer."""

    key: torch.Tensor
    value: torch.Tensor


class _RotaryEmbedding3D(nn.Module):
    """Relative 3D rotary embeddings for temporal and spatial coordinates."""

    def __init__(
        self,
        *,
        temporal_dim: int = 16,
        height_dim: int = 24,
        width_dim: int = 24,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        dimensions = (temporal_dim, height_dim, width_dim)
        if any(dimension <= 0 or dimension % 2 != 0 for dimension in dimensions):
            raise ValueError(f"3D-RoPE dimensions must be positive and even: {dimensions}.")
        if theta <= 0:
            raise ValueError(f"RoPE theta must be positive, got {theta}.")

        self.head_dim = sum(dimensions)
        self._frequency_dimensions = dimensions
        self._frequency_theta = float(theta)
        self.register_buffer(
            "temporal_inv_freq",
            self._build_inv_freq(temporal_dim, theta),
            persistent=False,
        )
        self.register_buffer(
            "height_inv_freq",
            self._build_inv_freq(height_dim, theta),
            persistent=False,
        )
        self.register_buffer(
            "width_inv_freq",
            self._build_inv_freq(width_dim, theta),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Module.to()/half()/bfloat16() also cast nonpersistent buffers. Rebuild
        # from the frequency definition, never by upcasting rounded frequencies.
        # This also materializes valid constants after meta -> to_empty().
        for name, dimension in zip(
            ("temporal_inv_freq", "height_inv_freq", "width_inv_freq"),
            self._frequency_dimensions,
            strict=True,
        ):
            self._buffers[name] = self._build_inv_freq(
                dimension, self._frequency_theta, device=self._buffers[name].device
            )
        return self

    @staticmethod
    def _build_inv_freq(dimension: int, theta: float, *, device=None) -> torch.Tensor:
        indices = torch.arange(0, dimension, 2, dtype=torch.float32, device=device)
        return 1.0 / (theta ** (indices / dimension))

    def frequencies(
        self,
        temporal_positions: torch.Tensor,
        height_positions: torch.Tensor,
        width_positions: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosines = []
        sines = []
        for positions, inv_freq in (
            (temporal_positions, self.temporal_inv_freq),
            (height_positions, self.height_inv_freq),
            (width_positions, self.width_inv_freq),
        ):
            angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
            cosines.append(torch.cos(angles).repeat_interleave(2, dim=-1))
            sines.append(torch.sin(angles).repeat_interleave(2, dim=-1))
        return (
            torch.cat(cosines, dim=-1).to(dtype=dtype),
            torch.cat(sines, dim=-1).to(dtype=dtype),
        )

    @staticmethod
    def apply(
        tensor: torch.Tensor,
        cosines: torch.Tensor,
        sines: torch.Tensor,
    ) -> torch.Tensor:
        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated = torch.empty_like(tensor)
        cosines = cosines.unsqueeze(0).unsqueeze(0)
        sines = sines.unsqueeze(0).unsqueeze(0)
        rotated[..., 0::2] = even * cosines[..., 0::2] - odd * sines[..., 0::2]
        rotated[..., 1::2] = even * sines[..., 1::2] + odd * cosines[..., 1::2]
        return rotated


class _SwiGLU(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int) -> None:
        super().__init__()

        self.gate_up_proj = nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )

        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


__all__ = ["_KVCache", "_RotaryEmbedding3D", "_SwiGLU"]
