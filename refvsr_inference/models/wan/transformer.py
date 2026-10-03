# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
# Modified for RelayVSR: streaming inference, sparse conditioning, and spatial attention.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models._modeling_parallel import (
    ContextParallelInput,
    ContextParallelOutput,
)
from diffusers.models.attention import (
    AttentionMixin,
    AttentionModuleMixin,
    FeedForward,
)
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
)
from torch.nn.attention.flex_attention import (
    flex_attention as _flex_attention,
)

from refvsr_inference.models.wan.spatial_attention import SpatialAttentionSpec, spatial_attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_compiled_flex_attention = None


@dataclass
class WanSelfAttentionKVCache:
    """Fixed-capacity ring containing normalized, entirely unrotated K and V."""

    window_size: int = 4
    key_value: tuple[torch.Tensor, torch.Tensor] | None = None
    tokens_per_frame: int | None = None
    next_frame: int = 0
    _storage: tuple[torch.Tensor, torch.Tensor] | None = None

    def update(
        self, key: torch.Tensor, value: torch.Tensor, frame_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frame_index != self.next_frame:
            raise ValueError("Streaming KV cache requires consecutive frame indices.")
        if key.shape != value.shape:
            raise ValueError("Streaming K/V shapes must match.")
        b, tokens, heads, dim = key.shape
        if self._storage is None:
            self.tokens_per_frame = tokens
            shape = (b, self.window_size * tokens, heads, dim)
            self._storage = key.new_empty(shape), value.new_empty(shape)
        cached_key, cached_value = self._storage
        if (
            self.tokens_per_frame != tokens
            or cached_key.shape[0] != b
            or cached_key.shape[2:] != key.shape[2:]
            or cached_key.device != key.device
            or cached_key.dtype != key.dtype
        ):
            raise ValueError("Streaming KV cache shape, device and dtype must remain constant.")
        start = (frame_index % self.window_size) * tokens
        cached_key[:, start : start + tokens].copy_(key.detach())
        cached_value[:, start : start + tokens].copy_(value.detach())
        valid_tokens = min(frame_index + 1, self.window_size) * tokens
        self.key_value = cached_key[:, :valid_tokens], cached_value[:, :valid_tokens]
        self.next_frame += 1
        return self.key_value


@dataclass(frozen=True)
class WanStreamingRotaryEmbedding:
    query: tuple[torch.Tensor, torch.Tensor]
    key: tuple[torch.Tensor, torch.Tensor]
    frame_index: int


@dataclass
class WanCrossAttentionKVCache:
    """Projected text keys and values shared by all matching streaming samples."""

    key_value: tuple[torch.Tensor, torch.Tensor] | None = None


@dataclass
class WanConditioningCache:
    """Static one-step conditioning reused across frames and input samples."""

    temb: torch.Tensor
    timestep_proj: torch.Tensor
    encoder_hidden_states: torch.Tensor
    cross_attention_caches: list[WanCrossAttentionKVCache]


@dataclass
class WanStreamingState:
    """Mutable state owned by one streaming timestep and one video batch."""

    block_caches: list[WanSelfAttentionKVCache]
    conditioning_cache: WanConditioningCache | None = None
    frame_index: int = 0
    grid_signature: tuple | None = None


def _apply_wan_rotary(hidden_states, rotary_emb):
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos, sin = freqs_cos[..., 0::2], freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _get_qkv_projections(
    attn: "WanAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


class WanAttnProcessor:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask: BlockMask | SpatialAttentionSpec | None = None,
        self_attn_cache: WanSelfAttentionKVCache | None = None,
        cross_attn_cache: WanCrossAttentionKVCache | None = None,
    ) -> torch.Tensor:
        if cross_attn_cache is None:
            query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
            key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
            value = value.unflatten(2, (attn.heads, -1))
        else:
            if encoder_hidden_states is None or not attn.is_cross_attention:
                raise ValueError(
                    "Cross-attention KV cache requires cross-attention encoder states."
                )
            if rotary_emb is not None or block_mask is not None:
                raise ValueError(
                    "Cross-attention KV cache does not support rotary embeddings or block masks."
                )
            query = attn.to_q(hidden_states)
            if cross_attn_cache.key_value is None:
                raise RuntimeError("Streaming cross-attention cache was not prepared.")
            key, value = cross_attn_cache.key_value

        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))

        if self_attn_cache is not None:
            if encoder_hidden_states is not None or attention_mask is not None:
                raise ValueError("Streaming cache requires unmasked self-attention.")
            if not isinstance(rotary_emb, WanStreamingRotaryEmbedding):
                raise ValueError("Streaming cache requires separate Q/K rotary positions.")
            key, value = self_attn_cache.update(key, value, rotary_emb.frame_index)
            query = _apply_wan_rotary(query, rotary_emb.query)
            key = _apply_wan_rotary(key, rotary_emb.key)
        elif rotary_emb is not None:
            query = _apply_wan_rotary(query, rotary_emb)
            key = _apply_wan_rotary(key, rotary_emb)

        if block_mask is not None:
            hidden_states = _run_flex_attention(query, key, value, attention_mask, block_mask)
        else:
            if (
                attention_mask is not None
                and attention_mask.ndim == 2
                and attention_mask.shape[0] == query.shape[0]
                and attention_mask.shape[1] == key.shape[1]
            ):
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

            hidden_states = F.scaled_dot_product_attention(
                query=query.transpose(1, 2),
                key=key.transpose(1, 2),
                value=value.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def _run_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    block_mask: BlockMask | SpatialAttentionSpec | None,
) -> torch.Tensor:
    if attention_mask is not None:
        raise ValueError(
            "flex_attention backend does not support attention_mask; use block_mask instead."
        )

    if isinstance(block_mask, SpatialAttentionSpec):
        # Native SDPA/Flex honors autocast; the custom Triton entry does not.
        # Normalization may return FP32 even when the projected V is BF16.
        return spatial_attention(query.to(value.dtype), key.to(value.dtype), value, block_mask)

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    original_query_length = query.shape[2]

    if block_mask is not None:
        query, key, value = _pad_qkv_for_block_mask(query, key, value, block_mask)

    if not query.is_cuda and torch.is_grad_enabled():
        raise RuntimeError("flex_attention backend only supports training on CUDA devices.")
    if query.is_cuda:
        attention_fn = _get_cuda_flex_attention()
        hidden_states = attention_fn(query=query, key=key, value=value, block_mask=block_mask)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="flex_attention called without torch.compile")
            hidden_states = _flex_attention(
                query=query, key=key, value=value, block_mask=block_mask
            )

    hidden_states = hidden_states[:, :, :original_query_length]
    return hidden_states.transpose(1, 2)


def _get_cuda_flex_attention():
    # Let an outer compiled transformer block capture FlexAttention directly.
    # The standalone compiled kernel remains the eager-model fallback.
    if torch.compiler.is_compiling():
        return _flex_attention
    return _get_compiled_flex_attention()


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(
            _flex_attention,
            dynamic=False,
            mode="max-autotune-no-cudagraphs",
        )
    return _compiled_flex_attention


def _pad_qkv_for_block_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: BlockMask,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_length, key_length = block_mask.seq_lengths
    query = _pad_sequence_dim(query, query_length)
    key = _pad_sequence_dim(key, key_length)
    value = _pad_sequence_dim(value, key_length)
    return query, key, value


def _pad_sequence_dim(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
    padding_length = target_length - hidden_states.shape[2]
    if padding_length < 0:
        raise ValueError(
            f"Block mask length {target_length} is shorter than attention sequence length {hidden_states.shape[2]}."
        )
    if padding_length == 0:
        return hidden_states

    padding_shape = list(hidden_states.shape)
    padding_shape[2] = padding_length
    padding = hidden_states.new_zeros(padding_shape)
    return torch.cat([hidden_states, padding], dim=2)


def _validate_attention_config(
    *,
    num_frames_per_block: int,
    local_attn_size: int,
    sink_size: int,
    flex_attention_block_size: int,
) -> None:
    if num_frames_per_block <= 0:
        raise ValueError(f"num_frames_per_block must be positive, got {num_frames_per_block}.")
    if local_attn_size < -1:
        raise ValueError(f"local_attn_size must be -1 or non-negative, got {local_attn_size}.")
    if sink_size < 0:
        raise ValueError(f"sink_size must be non-negative, got {sink_size}.")
    if flex_attention_block_size <= 0:
        raise ValueError(
            f"flex_attention_block_size must be positive, got {flex_attention_block_size}."
        )


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        )

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat(
                [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
            )
            concatenated_bias = torch.cat(
                [self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]
            )
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias},
                strict=True,
                assign=True,
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def precompute_cross_attention_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        cache: WanCrossAttentionKVCache,
    ) -> None:
        if not self.is_cross_attention:
            raise ValueError("Cross-attention KV precompute requires cross-attention.")
        if self.fused_projections:
            key, value = self.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        else:
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)
        key = self.norm_k(key).unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        cache.key_value = key.detach(), value.detach()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        block_mask: BlockMask | SpatialAttentionSpec | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            block_mask,
            **kwargs,
        )


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: int | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        return temb, timestep_proj, encoder_hidden_states


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def _apply(self, fn, recurse=True):
        # Model.to(dtype=bf16) must not quantize the RoPE frequency source.
        # Retain its precision while following the destination device.
        cos, sin = self.freqs_cos, self.freqs_sin
        super()._apply(fn, recurse=recurse)
        if not cos.is_meta:
            self.freqs_cos = cos.to(device=self.freqs_cos.device)
            self.freqs_sin = sin.to(device=self.freqs_sin.device)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        frame_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        del batch_size, num_channels
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        if frame_offset < 0:
            raise ValueError(f"frame_offset must be non-negative, got {frame_offset}.")
        if frame_offset + ppf > self.max_seq_len:
            raise ValueError(
                "Temporal rotary position exceeds rope_max_seq_len: "
                f"offset={frame_offset}, frames={ppf}, max={self.max_seq_len}."
            )

        return self.for_positions(pph, ppw, tuple(range(frame_offset, frame_offset + ppf)))

    def for_positions(self, pph: int, ppw: int, positions: tuple[int, ...]):
        if not positions or min(positions) < 0 or max(positions) >= self.max_seq_len:
            raise ValueError("Temporal rotary positions exceed rope_max_seq_len.")
        if max(pph, ppw) > self.max_seq_len:
            raise ValueError("Spatial rotary positions exceed rope_max_seq_len.")
        ppf = len(positions)
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        temporal_slice = torch.tensor(positions, device=self.freqs_cos.device)
        freqs_cos_f = freqs_cos[0][temporal_slice].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][temporal_slice].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(
            1, ppf * pph * ppw, 1, -1
        )

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        block_mask: BlockMask | SpatialAttentionSpec | None = None,
        self_attn_cache: WanSelfAttentionKVCache | None = None,
        cross_attn_cache: WanCrossAttentionKVCache | None = None,
        precompute_cross_attention_only: bool = False,
    ) -> torch.Tensor:
        if precompute_cross_attention_only:
            if cross_attn_cache is None:
                raise ValueError("Cross-attention precompute requires a cache.")
            self.attn2.precompute_cross_attention_kv(
                encoder_hidden_states,
                cross_attn_cache,
            )
            return hidden_states

        assert temb.ndim == 4
        # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        # batch_size, seq_len, 1, inner_dim
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)

        # 1. Self-attention
        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            None,
            rotary_emb,
            block_mask,
            self_attn_cache=self_attn_cache,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            None,
            None,
            cross_attn_cache=cross_attn_cache,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(
            hidden_states
        )

        return hidden_states


class WanTransformer3DModel(
    ModelMixin,
    ConfigMixin,
    PeftAdapterMixin,
    FromOriginalModelMixin,
    CacheMixin,
    AttentionMixin,
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        # Reference: https://github.com/huggingface/diffusers/pull/12909
        # We need to disable the splitting of encoder_hidden_states because the image_encoder
        # (Wan 2.1 I2V) consistently generates 257 tokens for image_embed. This causes the shape
        # of encoder_hidden_states—whose token count is always 769 (512 + 257) after concatenation
        # —to be indivisible by the number of devices in the CP.
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
        lq_proj_hidden_dim1: int = 2048,
        lq_proj_hidden_dim2: int = 3072,
        use_block_causal_attention: bool = False,
        num_frames_per_block: int = 1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        flex_attention_block_size: int = 128,
        spatial_window_size: tuple[int, int] | None = None,
        spatial_attention_backend: str = "flex",
        spatial_attention_mode: str | None = None,
        temporal_context_frames: int | None = None,
        temporal_rope: dict | None = None,
    ) -> None:
        super().__init__()
        if temporal_context_frames is not None:
            if temporal_context_frames <= 0:
                raise ValueError("temporal_context_frames must be positive.")
            if local_attn_size not in (-1, temporal_context_frames):
                raise ValueError("Conflicting temporal_context_frames and local_attn_size.")
        if spatial_attention_mode not in (None, "global", "local"):
            raise ValueError("spatial_attention_mode must be global or local.")
        if spatial_attention_mode == "local" and spatial_window_size is None:
            raise ValueError("local spatial_attention_mode requires spatial_window_size.")
        rope_config = dict(temporal_rope or {})
        if set(rope_config) - {"policy", "training_frames"}:
            raise ValueError("Unknown temporal_rope option.")
        if rope_config.get("policy", "fixed_end") != "fixed_end":
            raise ValueError("Streaming temporal_rope.policy must be fixed_end.")
        if int(rope_config.get("training_frames", 8)) <= 0:
            raise ValueError("temporal_rope.training_frames must be positive.")
        if spatial_attention_backend not in ("flex", "triton"):
            raise ValueError("spatial_attention_backend must be flex or triton")
        if spatial_window_size is not None:
            if len(spatial_window_size) != 2 or any(int(v) <= 0 for v in spatial_window_size):
                raise ValueError("spatial_window_size must contain two positive token counts.")
            if not use_block_causal_attention:
                raise ValueError("Spatial locality requires use_block_causal_attention=True.")
        _validate_attention_config(
            num_frames_per_block=num_frames_per_block,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            flex_attention_block_size=flex_attention_block_size,
        )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(
            in_channels, inner_dim, kernel_size=patch_size, stride=patch_size
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        self._block_causal_mask_cache: dict[tuple, BlockMask] = {}

    @property
    def temporal_context_frames(self) -> int:
        configured = self.config.temporal_context_frames
        return int(self.config.local_attn_size if configured is None else configured)

    @property
    def spatial_window(self):
        if self.config.spatial_attention_mode == "global":
            return None
        return self.config.spatial_window_size

    def _get_block_causal_attention_mask(
        self,
        *,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        frame_height: int | None = None,
        frame_width: int | None = None,
    ) -> BlockMask | SpatialAttentionSpec | None:
        # A single frame has no temporal future to mask. Returning None routes
        # self-attention through SDPA and avoids Flex Attention's block-mask
        # construction, while later video batches still build and reuse masks.
        spatial_window = self.spatial_window
        if spatial_window is not None:
            if frame_height is None or frame_width is None:
                raise ValueError("Spatial locality requires the actual token height and width.")
            if frame_height * frame_width != frame_seqlen:
                raise ValueError("Token height and width do not match frame_seqlen.")
        spatial_is_global = spatial_window is None or (
            frame_height <= spatial_window[0] and frame_width <= spatial_window[1]
        )
        if not self.config.use_block_causal_attention or (num_frames == 1 and spatial_is_global):
            return None
        if spatial_is_global:
            # Reuse the identical temporal mask/kernel when the spatial window
            # covers the image. Redundant predicates can change BF16 autotuning.
            spatial_window = None

        if (
            spatial_window is not None
            and self.config.spatial_attention_backend == "triton"
            and not self.training
            and not torch.is_grad_enabled()
        ):
            if (
                device.type != "cuda"
                or self.config.num_frames_per_block != 1
                or self.config.sink_size != 0
                or self.temporal_context_frames <= 0
                or self.config.attention_head_dim not in (32, 64, 128)
            ):
                raise ValueError(
                    "Triton spatial inference requires CUDA, head dim 32/64/128, "
                    "one frame per block, no sink and a positive temporal window."
                )
            return SpatialAttentionSpec(
                num_frames,
                frame_height,
                frame_width,
                spatial_window[0],
                spatial_window[1],
                self.temporal_context_frames,
            )

        cache_key = self._block_causal_mask_cache_key(device, num_frames, frame_seqlen) + (
            frame_height,
            frame_width,
            tuple(spatial_window) if spatial_window is not None else None,
        )
        block_mask = self._block_causal_mask_cache.get(cache_key)
        if block_mask is None:
            block_mask = self._prepare_block_causal_attention_mask(
                device=device,
                num_frames=num_frames,
                frame_seqlen=frame_seqlen,
                num_frames_per_block=int(self.config.num_frames_per_block),
                local_attn_size=self.temporal_context_frames,
                sink_size=int(self.config.sink_size),
                flex_attention_block_size=int(self.config.flex_attention_block_size),
                spatial_window_size=spatial_window,
                frame_height=frame_height,
                frame_width=frame_width,
            )
            self._block_causal_mask_cache[cache_key] = block_mask
            logger.info(
                "Cached block causal attention mask: frames=%s, frame_seqlen=%s, frames_per_block=%s",
                num_frames,
                frame_seqlen,
                self.config.num_frames_per_block,
            )
        return block_mask

    def _block_causal_mask_cache_key(
        self,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
    ) -> tuple:
        device = torch.device(device)
        return (
            device.type,
            device.index,
            int(num_frames),
            int(frame_seqlen),
            int(self.config.num_frames_per_block),
            self.temporal_context_frames,
            int(self.config.sink_size),
            int(self.config.flex_attention_block_size),
        )

    @staticmethod
    def _prepare_block_causal_attention_mask(
        *,
        device: torch.device,
        num_frames: int,
        frame_seqlen: int,
        num_frames_per_block: int = 1,
        local_attn_size: int = -1,
        sink_size: int = 0,
        flex_attention_block_size: int = 128,
        spatial_window_size: tuple[int, int] | None = None,
        frame_height: int | None = None,
        frame_width: int | None = None,
    ) -> BlockMask:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}.")
        if frame_seqlen <= 0:
            raise ValueError(f"frame_seqlen must be positive, got {frame_seqlen}.")

        if spatial_window_size is not None:
            if (
                frame_height is None
                or frame_width is None
                or frame_height * frame_width != frame_seqlen
            ):
                raise ValueError("Spatial mask requires matching token height and width.")
            if len(spatial_window_size) != 2 or min(spatial_window_size) <= 0:
                raise ValueError("spatial_window_size must contain two positive token counts.")
            window_height = min(int(spatial_window_size[0]), frame_height)
            window_width = min(int(spatial_window_size[1]), frame_width)

        total_length = num_frames * frame_seqlen
        padded_length = (
            math.ceil(total_length / flex_attention_block_size) * flex_attention_block_size
        )
        block_token_length = frame_seqlen * num_frames_per_block
        token_indices = torch.arange(padded_length, device=device, dtype=torch.long)
        block_ends = ((token_indices // block_token_length) + 1) * block_token_length
        block_ends = block_ends.clamp(max=total_length)
        sink_token_length = sink_size * frame_seqlen
        local_window_length = local_attn_size * frame_seqlen

        def block_causal_mask(_batch, _head, query_index, key_index):
            is_valid_query = query_index < total_length
            same_padding_token = query_index == key_index
            is_block_causal = is_valid_query & (key_index < block_ends[query_index])

            if local_attn_size == -1:
                allowed = is_block_causal
            else:
                in_window = key_index >= (block_ends[query_index] - local_window_length)
                in_sink = key_index < sink_token_length
                allowed = is_block_causal & (in_window | in_sink)

            if spatial_window_size is not None:
                # Coordinates are on the original DiT grid, before block padding.
                # Shift the rectangular neighborhood inward at image boundaries.
                query_y = (query_index % frame_seqlen) // frame_width
                query_x = query_index % frame_width
                key_y = (key_index % frame_seqlen) // frame_width
                key_x = key_index % frame_width
                top = (query_y - window_height // 2).clamp(0, frame_height - window_height)
                left = (query_x - window_width // 2).clamp(0, frame_width - window_width)
                in_space = (
                    (key_y >= top)
                    & (key_y < top + window_height)
                    & (key_x >= left)
                    & (key_x < left + window_width)
                )
                allowed = allowed & in_space
            return allowed | same_padding_token

        return create_block_mask(
            block_causal_mask,
            B=None,
            H=None,
            Q_LEN=padded_length,
            KV_LEN=padded_length,
            BLOCK_SIZE=flex_attention_block_size,
            _compile=False,
            device=device,
        )

    def build_lq_proj(self):
        from .lq_proj import LQ4xProj

        return LQ4xProj(
            in_dim=3,
            out_dim=self.config.num_attention_heads * self.config.attention_head_dim,
            hidden_dim1=self.config.lq_proj_hidden_dim1,
            hidden_dim2=self.config.lq_proj_hidden_dim2,
        )

    def create_streaming_state(
        self,
        *,
        kv_cache_window_size: int = -1,
        conditioning_cache: WanConditioningCache | None = None,
    ) -> WanStreamingState:
        """Create an empty KV cache for one timestep and video batch."""
        if not self.config.use_block_causal_attention:
            raise ValueError("Streaming inference requires use_block_causal_attention=True.")
        if int(self.config.num_frames_per_block) != 1:
            raise ValueError("Streaming inference requires num_frames_per_block=1.")
        if int(self.config.sink_size) != 0:
            raise ValueError("Streaming inference requires sink_size=0.")
        if kv_cache_window_size == 0 or kv_cache_window_size < -1:
            raise ValueError("kv_cache_window_size must be -1 or a positive integer.")
        window = self.temporal_context_frames
        if kv_cache_window_size > 0:
            if window > 0 and window != kv_cache_window_size:
                raise ValueError("Conflicting temporal_context_frames and kv_cache_window_size.")
            window = kv_cache_window_size
        if window == -1:
            window = 4
        training_frames = int((self.config.temporal_rope or {}).get("training_frames", 8))
        if not 1 <= window <= training_frames <= self.rope.max_seq_len:
            raise ValueError(
                "Streaming requires 1 <= temporal_context_frames <= training_frames <= rope_max_seq_len."
            )
        return WanStreamingState(
            block_caches=[WanSelfAttentionKVCache(window_size=window) for _ in self.blocks],
            conditioning_cache=conditioning_cache,
        )

    def _embed_conditioning(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if timestep.ndim == 2:
            timestep_seq_len = timestep.shape[1]
            flattened_timestep = timestep.flatten()
        else:
            timestep_seq_len = None
            flattened_timestep = timestep
        temb, timestep_proj, projected_encoder_hidden_states = self.condition_embedder(
            flattened_timestep,
            encoder_hidden_states,
            timestep_seq_len=timestep_seq_len,
        )
        if timestep_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))
        return temb, timestep_proj, projected_encoder_hidden_states

    def _cache_streaming_conditioning(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> WanConditioningCache:
        if encoder_hidden_states.shape[0] != 1:
            raise ValueError("Streaming conditioning requires prompt batch size 1.")
        if timestep.ndim == 2:
            base_timestep = timestep[:, :1]
        elif timestep.ndim == 1:
            base_timestep = timestep[:, None]
        else:
            raise ValueError("Streaming timestep must have shape [batch] or [batch, tokens].")
        if base_timestep.shape[0] != 1:
            raise ValueError("Streaming conditioning requires timestep batch size 1.")

        temb, timestep_proj, projected_encoder_hidden_states = self.condition_embedder(
            base_timestep.flatten(),
            encoder_hidden_states,
            timestep_seq_len=1,
        )
        cache = WanConditioningCache(
            temb=temb.detach(),
            timestep_proj=timestep_proj.unflatten(2, (6, -1)).detach(),
            encoder_hidden_states=projected_encoder_hidden_states.detach(),
            cross_attention_caches=[WanCrossAttentionKVCache() for _ in self.blocks],
        )
        return cache

    def _expand_streaming_conditioning(
        self,
        *,
        streaming_state: WanStreamingState,
        batch_size: int,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = streaming_state.conditioning_cache
        if cache is None:
            raise RuntimeError(
                "Streaming conditioning was not prepared; call "
                "pipeline.prepare_streaming() before sampling."
            )
        return (
            cache.temb.expand(batch_size, token_count, -1),
            cache.timestep_proj.expand(batch_size, token_count, -1, -1),
            cache.encoder_hidden_states.expand(batch_size, -1, -1),
        )

    def _precompute_streaming(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        streaming_state: WanStreamingState | None,
    ) -> torch.Tensor:
        """Populate streaming conditioning caches through transformer block forwards."""
        if streaming_state is None:
            raise ValueError("Streaming precompute requires a streaming state.")
        batch_size, _, num_frames, _, _ = hidden_states.shape
        if num_frames != 1 or self.config.patch_size[0] != 1:
            raise ValueError(
                "Streaming precompute requires a one-frame latent and temporal patch_size=1."
            )

        cache = self._cache_streaming_conditioning(
            timestep,
            encoder_hidden_states,
        )
        streaming_state.conditioning_cache = cache
        empty_hidden_states = cache.encoder_hidden_states.new_empty(
            (
                batch_size,
                0,
                self.config.num_attention_heads * self.config.attention_head_dim,
            )
        )
        for index, block in enumerate(self.blocks):
            empty_hidden_states = block(
                empty_hidden_states,
                cache.encoder_hidden_states,
                cache.timestep_proj,
                (self.rope.freqs_cos, self.rope.freqs_sin),
                cross_attn_cache=(cache.cross_attention_caches[index]),
                precompute_cross_attention_only=True,
            )
        return hidden_states

    def forward(
        self,
        lq_videos: torch.Tensor | None,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        extract_layers: Sequence[int] | None = None,
        use_lq_condition: bool = True,
        streaming_state: WanStreamingState | None = None,
        precompute_streaming: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Run full-sequence, one-frame streaming, or streaming precompute."""
        if precompute_streaming:
            return self._precompute_streaming(
                hidden_states,
                timestep,
                encoder_hidden_states,
                streaming_state,
            )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        del num_channels
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        if streaming_state is not None:
            if extract_layers is not None:
                raise ValueError("extract_layers is not supported during streaming inference.")
            if num_frames != 1 or p_t != 1:
                raise ValueError(
                    "A streaming forward call must contain exactly one frame and use temporal patch_size=1."
                )
            if len(streaming_state.block_caches) != len(self.blocks):
                raise ValueError("Streaming state block count does not match the transformer.")
            signature = (
                batch_size,
                height,
                width,
                hidden_states.device,
                hidden_states.dtype,
            )
            if streaming_state.grid_signature is None:
                streaming_state.grid_signature = signature
            elif streaming_state.grid_signature != signature:
                raise ValueError(
                    "Streaming grid, batch, device and dtype must remain constant; create a new state."
                )
            frame = streaming_state.frame_index
            window = streaming_state.block_caches[0].window_size
            valid_frames = min(frame + 1, window)
            training_frames = int((self.config.temporal_rope or {}).get("training_frames", 8))
            origin = max(0, frame - training_frames + 1)
            # Physical slots are stable; RoPE follows the true frame occupying each slot.
            frame_ids = tuple(frame - ((frame - slot) % window) for slot in range(valid_frames))
            rotary_emb = WanStreamingRotaryEmbedding(
                query=self.rope.for_positions(
                    post_patch_height, post_patch_width, (frame - origin,)
                ),
                key=self.rope.for_positions(
                    post_patch_height,
                    post_patch_width,
                    tuple(index - origin for index in frame_ids),
                ),
                frame_index=frame,
            )
        else:
            rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        if use_lq_condition:
            if lq_videos is None:
                raise ValueError("lq_videos is required when use_lq_condition=True.")
            hidden_states = hidden_states + self.lq_proj(lq_videos)
        block_mask = None
        if streaming_state is None:
            block_mask = self._get_block_causal_attention_mask(
                device=hidden_states.device,
                num_frames=post_patch_num_frames,
                frame_seqlen=post_patch_height * post_patch_width,
                frame_height=post_patch_height,
                frame_width=post_patch_width,
            )

        if streaming_state is not None and self.spatial_window is not None:
            wh, ww = self.spatial_window
            if wh < post_patch_height or ww < post_patch_width:
                spec = SpatialAttentionSpec(
                    1,
                    post_patch_height,
                    post_patch_width,
                    wh,
                    ww,
                    window,
                    key_frames=valid_frames,
                    query_frame_offset=valid_frames - 1,
                )
                if hidden_states.is_cuda and self.config.spatial_attention_backend == "triton":
                    block_mask = spec
                else:
                    block_mask = create_block_mask(
                        spec.mask_mod(),
                        B=None,
                        H=None,
                        Q_LEN=post_patch_height * post_patch_width,
                        KV_LEN=valid_frames * post_patch_height * post_patch_width,
                        device=str(hidden_states.device),
                    )

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if streaming_state is None:
            temb, timestep_proj, encoder_hidden_states = self._embed_conditioning(
                timestep,
                encoder_hidden_states,
            )
        else:
            if timestep.ndim == 2:
                token_count = timestep.shape[1]
            elif timestep.ndim == 1:
                token_count = 1
            else:
                raise ValueError("Streaming timestep must have shape [batch] or [batch, tokens].")
            temb, timestep_proj, encoder_hidden_states = self._expand_streaming_conditioning(
                streaming_state=streaming_state,
                batch_size=batch_size,
                token_count=token_count,
            )

        # 4. Transformer blocks
        extract_layer_set = set(extract_layers or ())
        last_extract_layer = max(extract_layer_set, default=None)
        extracted_hidden_states: list[torch.Tensor] = []
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            if streaming_state is not None:
                raise RuntimeError(
                    "Streaming inference is incompatible with gradient checkpointing."
                )
            for index, block in enumerate(self.blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    block_mask,
                )
                if index in extract_layer_set:
                    extracted_hidden_states.append(hidden_states)
                if last_extract_layer is not None and index >= last_extract_layer:
                    break
        else:
            for index, block in enumerate(self.blocks):
                self_attn_cache = (
                    streaming_state.block_caches[index] if streaming_state is not None else None
                )
                cross_attn_cache = None
                if streaming_state is not None:
                    assert streaming_state.conditioning_cache is not None
                    cross_attn_cache = streaming_state.conditioning_cache.cross_attention_caches[
                        index
                    ]
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep_proj,
                    rotary_emb,
                    block_mask,
                    self_attn_cache,
                    cross_attn_cache,
                )
                if index in extract_layer_set:
                    extracted_hidden_states.append(hidden_states)
                if last_extract_layer is not None and index >= last_extract_layer:
                    break

        if extract_layers is not None:
            return extracted_hidden_states

        # 5. Output norm, projection & unpatchify
        assert temb.ndim == 3
        # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
        shift, scale = (
            self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)
        ).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
            hidden_states
        )
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if streaming_state is not None:
            streaming_state.frame_index += post_patch_num_frames

        return output
