"""FlashDecoder-style streaming VSR with configurable persistent conditions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import nn

from refvsr_inference.models.flash._flashvsr_ops import (
    _KVCache,
    _RotaryEmbedding3D,
    _SwiGLU,
)

from ._local_attention import LocalAttentionConfig, local_attention

FlashVSRConditionVariant = Literal["halfS", "S", "M", "B", "L", "XL"]
FlashVSRGradientCheckpointing = Literal["none", "mlp", "block"]
FlashVSRConditionMode = Literal["start", "endpoints"]


@dataclass(frozen=True)
class FlashVSRConditionState:
    """Unrotated KV with frame_index measured from this stream/window's start."""

    video: tuple[_KVCache | None, ...]
    condition: tuple[_KVCache | None, ...]
    frame_index: int = 0
    batch_size: int | None = None
    input_height: int | None = None
    input_width: int | None = None
    total_frames: int | None = None
    spatial_attention: LocalAttentionConfig | None = None
    primed: bool = False
    end_condition_keep_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class FlashVSRPreparedCondition:
    """Unrotated endpoint KV; its start/end RoPE position is assigned on use."""

    caches: tuple[_KVCache, ...]
    batch_size: int
    height: int
    width: int
    owner: int


@dataclass(frozen=True)
class _VariantSpec:
    depth: int
    dim: int
    num_heads: int
    num_kv_groups: int


class _LRFrameEncoder(nn.Module):
    """Encode an LR frame into normalized tokens at 1/4 resolution."""

    def __init__(
        self,
        *,
        dim: int,
        channels: tuple[int, int, int],
        norm_eps: float,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError(
                f"LR encoder channels must contain three positive values, got {channels}."
            )

        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels, stride in zip(channels, (2, 2, 1), strict=True):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    ),
                    nn.SiLU(),
                ]
            )
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels, dim, kernel_size=1))
        self.layers = nn.Sequential(*layers)
        self.norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, lr_frame: torch.Tensor) -> torch.Tensor:
        tokens = self.layers(lr_frame).flatten(2).transpose(1, 2)
        return self.norm(tokens)


class _ConditionEncoder(nn.Module):
    """Map a normalized Wan latent onto the shared VSR token space."""

    def __init__(self, *, latent_channels: int, dim: int, norm_eps: float) -> None:
        super().__init__()
        self.projection = nn.Conv2d(latent_channels, dim, kernel_size=1)
        self.norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        tokens = self.projection(latent).flatten(2).transpose(1, 2)
        return self.norm(tokens)


class _ConditionedGroupedQueryAttention(nn.Module):
    """GQA over persistent condition, rolling video, and current video KV."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        norm_eps: float,
        spatial_attention: LocalAttentionConfig | None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.spatial_attention = spatial_attention
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = dim // num_heads
        self.kv_dim = num_kv_groups * self.head_dim

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, self.kv_dim)
        self.value = nn.Linear(dim, self.kv_dim)
        self.condition_key = nn.Linear(dim, self.kv_dim)
        self.condition_value = nn.Linear(dim, self.kv_dim)
        self.key_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.value_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
        self.output = nn.Linear(dim, dim)

    def prime_condition(
        self,
        hidden_states: torch.Tensor,
        *,
        rope: tuple[torch.Tensor, torch.Tensor],
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, _KVCache]:
        query, key, value = self._project(
            hidden_states,
            key_projection=self.condition_key,
            value_projection=self.condition_value,
        )
        cache = _KVCache(key=key, value=value)
        attended = self._attend(
            query,
            key,
            value,
            query_rope=rope,
            key_rope=rope,
            grid_size=grid_size,
            condition_only=True,
        )
        return attended, cache

    def forward_video(
        self,
        hidden_states: torch.Tensor,
        *,
        video_cache: _KVCache | None,
        condition_cache: _KVCache,
        query_rope: tuple[torch.Tensor, torch.Tensor],
        key_rope: tuple[torch.Tensor, torch.Tensor],
        grid_size: tuple[int, int],
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, _KVCache]:
        query, current_key, current_value = self._project(hidden_states)
        if video_cache is None:
            video_key = current_key
            video_value = current_value
        else:
            video_key = torch.cat((video_cache.key, current_key), dim=2)
            video_value = torch.cat((video_cache.value, current_value), dim=2)

        next_video_cache = _KVCache(key=video_key, value=video_value)
        key = torch.cat((condition_cache.key, video_key), dim=2)
        value = torch.cat((condition_cache.value, video_value), dim=2)
        attended = self._attend(
            query,
            key,
            value,
            query_rope=query_rope,
            key_rope=key_rope,
            grid_size=grid_size,
            end_condition_keep_mask=end_condition_keep_mask,
            condition_planes=condition_cache.key.shape[2] // (grid_size[0] * grid_size[1]),
        )
        return attended, next_video_cache

    def _project(
        self,
        hidden_states: torch.Tensor,
        *,
        key_projection: nn.Linear | None = None,
        value_projection: nn.Linear | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = hidden_states.shape
        key_projection = self.key if key_projection is None else key_projection
        value_projection = self.value if value_projection is None else value_projection
        query = self.query(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )
        query = query.transpose(1, 2)

        key = key_projection(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_kv_groups,
            self.head_dim,
        )
        key = self.key_norm(key.transpose(1, 2))

        value = value_projection(hidden_states).view(
            batch_size,
            num_tokens,
            self.num_kv_groups,
            self.head_dim,
        )
        value = self.value_norm(value.transpose(1, 2))
        return query, key, value

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        query_rope: tuple[torch.Tensor, torch.Tensor],
        key_rope: tuple[torch.Tensor, torch.Tensor],
        grid_size: tuple[int, int],
        condition_only: bool = False,
        condition_planes: int = 0,
        parallel_frames: int | None = None,
        temporal_window: int = 1,
        relative_rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, num_tokens, _ = query.shape
        query = _RotaryEmbedding3D.apply(query, *query_rope)
        key = _RotaryEmbedding3D.apply(key, *key_rope)
        kv_plane_mask = None
        if end_condition_keep_mask is not None:
            if condition_only or condition_planes != 2:
                raise ValueError("End-condition masking requires two condition planes.")
            plane_ids = torch.arange(
                key.shape[2] // (grid_size[0] * grid_size[1]), device=query.device
            )
            kv_plane_mask = (plane_ids[None, :] != 1) | end_condition_keep_mask[:, None]
        if self.spatial_attention is None and parallel_frames is None:
            attended = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=(
                    kv_plane_mask.repeat_interleave(grid_size[0] * grid_size[1], dim=1)[
                        :, None, None, :
                    ]
                    if kv_plane_mask is not None
                    else None
                ),
                dropout_p=0.0,
                is_causal=False,
                enable_gqa=True,
            )
        else:
            height, width = grid_size
            config = self.spatial_attention or LocalAttentionConfig(
                video=(2 * height, 2 * width),
                start=(2 * height, 2 * width),
                end=(2 * height, 2 * width),
                condition_self=(2 * height, 2 * width),
                backend="sdpa",
            )
            if condition_only:
                windows = (config.condition_self,)
            else:
                video_planes = key.shape[2] // (height * width) - condition_planes
                windows = (config.start, config.end)[:condition_planes] + (
                    config.video,
                ) * video_planes
            attended = local_attention(
                query,
                key,
                value,
                height=height,
                width=width,
                windows=windows,
                backend=config.backend,
                query_frames=parallel_frames or 1,
                condition_planes=condition_planes if parallel_frames is not None else None,
                temporal_window=temporal_window,
                kv_plane_mask=kv_plane_mask,
                relative_rope=relative_rope,
            )
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            self.dim,
        )
        return self.output(attended)


class _ConditionedTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        hidden_dim: int,
        norm_eps: float,
        gradient_checkpointing: FlashVSRGradientCheckpointing,
        spatial_attention: LocalAttentionConfig | None,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.attention = _ConditionedGroupedQueryAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            norm_eps=norm_eps,
            spatial_attention=spatial_attention,
        )
        self.feed_forward_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.feed_forward = _SwiGLU(dim=dim, hidden_dim=hidden_dim)
        self.gradient_checkpointing = gradient_checkpointing

    def prime_condition(
        self,
        hidden_states: torch.Tensor,
        *,
        rope: tuple[torch.Tensor, torch.Tensor],
        grid_size: tuple[int, int],
    ) -> tuple[torch.Tensor, _KVCache]:
        attended, cache = self.attention.prime_condition(
            self.attention_norm(hidden_states),
            rope=rope,
            grid_size=grid_size,
        )
        hidden_states = hidden_states + attended
        hidden_states = self._apply_feed_forward(hidden_states)
        return hidden_states, cache

    def forward_video(
        self,
        hidden_states: torch.Tensor,
        *,
        video_cache: _KVCache | None,
        condition_cache: _KVCache,
        query_rope: tuple[torch.Tensor, torch.Tensor],
        key_rope: tuple[torch.Tensor, torch.Tensor],
        grid_size: tuple[int, int],
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, _KVCache]:
        attended, cache = self.attention.forward_video(
            self.attention_norm(hidden_states),
            video_cache=video_cache,
            condition_cache=condition_cache,
            query_rope=query_rope,
            key_rope=key_rope,
            grid_size=grid_size,
            end_condition_keep_mask=end_condition_keep_mask,
        )
        hidden_states = hidden_states + attended
        hidden_states = self._apply_feed_forward(hidden_states)
        return hidden_states, cache

    def _apply_feed_forward(self, hidden_states):
        return hidden_states + self._feed_forward_residual(hidden_states)

    def _feed_forward_residual(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.feed_forward(self.feed_forward_norm(hidden_states))


class _ConditionedStreamingTransformer(nn.Module):
    """Streaming stack with one or two fixed condition caches."""

    def __init__(
        self,
        *,
        depth: int,
        dim: int,
        num_heads: int,
        num_kv_groups: int,
        hidden_dim: int,
        max_video_frames: int,
        rope: _RotaryEmbedding3D,
        norm_eps: float,
        gradient_checkpointing: FlashVSRGradientCheckpointing,
        spatial_attention: LocalAttentionConfig | None,
    ) -> None:
        super().__init__()
        self.max_video_frames = max_video_frames
        self.rope = rope
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList(
            [
                _ConditionedTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    num_kv_groups=num_kv_groups,
                    hidden_dim=hidden_dim,
                    norm_eps=norm_eps,
                    gradient_checkpointing=gradient_checkpointing,
                    spatial_attention=spatial_attention,
                )
                for _ in range(depth)
            ]
        )

    def prime_condition(
        self,
        hidden_states: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, tuple[_KVCache, ...]]:
        rope = self._rope_for_frame(
            temporal_position=0,
            height=height,
            width=width,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        grid_size = (height, width)
        caches = []
        for block in self.blocks:
            hidden_states, cache = block.prime_condition(
                hidden_states,
                rope=rope,
                grid_size=grid_size,
            )
            caches.append(cache)
        return hidden_states, tuple(caches)

    def forward_video(
        self,
        hidden_states: torch.Tensor,
        *,
        video_caches: tuple[_KVCache | None, ...],
        condition_caches: tuple[_KVCache | None, ...],
        frame_index: int,
        condition_temporal_positions: tuple[int, ...],
        height: int,
        width: int,
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[_KVCache, ...]]:

        grid_size = (height, width)
        tokens_per_frame = height * width
        keep_tokens = (self.max_video_frames - 1) * tokens_per_frame
        video_caches = tuple(self._trim_video_cache(cache, keep_tokens) for cache in video_caches)
        previous_tokens = self._cache_length(video_caches)
        if previous_tokens % tokens_per_frame != 0:
            raise ValueError(
                f"Video cache length {previous_tokens} is not divisible by "
                f"tokens per frame {tokens_per_frame}."
            )
        previous_frames = previous_tokens // tokens_per_frame
        condition_tokens = self._cache_length(condition_caches)
        expected_condition_tokens = len(condition_temporal_positions) * tokens_per_frame
        if condition_tokens != expected_condition_tokens:
            raise ValueError(
                "Condition cache length does not match its temporal positions: "
                f"{condition_tokens} != {expected_condition_tokens}."
            )
        query_rope, key_rope = self._build_video_rope(
            frame_index=frame_index,
            previous_frames=previous_frames,
            condition_temporal_positions=condition_temporal_positions,
            height=height,
            width=width,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        next_video_caches = []
        for block, video_cache, condition_cache in zip(
            self.blocks,
            video_caches,
            condition_caches,
            strict=True,
        ):
            assert condition_cache is not None
            hidden_states, next_video_cache = block.forward_video(
                hidden_states,
                video_cache=video_cache,
                condition_cache=condition_cache,
                query_rope=query_rope,
                key_rope=key_rope,
                grid_size=grid_size,
                end_condition_keep_mask=end_condition_keep_mask,
            )
            next_video_caches.append(next_video_cache)
        return hidden_states, tuple(next_video_caches)

    def _use_block_checkpointing(self):
        return False

    def _build_video_rope(
        self,
        *,
        frame_index: int,
        previous_frames: int,
        condition_temporal_positions: tuple[int, ...],
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        plans = getattr(self, "_inference_rope_plans", None)
        cache_key = (
            height,
            width,
            str(device),
            dtype,
            previous_frames,
            len(condition_temporal_positions),
        )
        use_cache = plans is not None and not self.training and not torch.is_grad_enabled()
        if use_cache and cache_key in plans:
            return plans[cache_key]
        query_rope = self._rope_for_frame(
            temporal_position=0,
            height=height,
            width=width,
            device=device,
            dtype=dtype,
        )
        condition_t, condition_h, condition_w = self._token_positions(
            temporal_positions=torch.zeros(
                len(condition_temporal_positions), device=device, dtype=torch.long
            ),
            height=height,
            width=width,
        )
        condition_rope = self.rope.frequencies(
            condition_t,
            condition_h,
            condition_w,
            dtype=dtype,
        )
        # Unrotated cached keys are re-positioned relative to the current query.
        video_positions = torch.arange(-previous_frames, 1, device=device)
        video_t, video_h, video_w = self._token_positions(
            temporal_positions=video_positions,
            height=height,
            width=width,
        )
        video_rope = self.rope.frequencies(
            video_t,
            video_h,
            video_w,
            dtype=dtype,
        )
        key_rope = tuple(
            torch.cat((condition_part, video_part), dim=0)
            for condition_part, video_part in zip(
                condition_rope,
                video_rope,
                strict=True,
            )
        )
        if use_cache:
            if len(plans) >= 128:
                plans.pop(next(iter(plans)))
            plans[cache_key] = (query_rope, key_rope)
        return query_rope, key_rope

    def _rope_for_frame(
        self,
        *,
        temporal_position: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal, rows, columns = self._token_positions(
            temporal_positions=torch.tensor([temporal_position], device=device),
            height=height,
            width=width,
        )
        return self.rope.frequencies(
            temporal,
            rows,
            columns,
            dtype=dtype,
        )

    @staticmethod
    def _token_positions(
        *,
        temporal_positions: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = torch.arange(height, device=temporal_positions.device)
        columns = torch.arange(width, device=temporal_positions.device)
        temporal, rows, columns = torch.meshgrid(
            temporal_positions,
            rows,
            columns,
            indexing="ij",
        )
        return temporal.flatten(), rows.flatten(), columns.flatten()

    @staticmethod
    def _trim_video_cache(
        cache: _KVCache | None,
        keep_tokens: int,
    ) -> _KVCache | None:
        if cache is None or keep_tokens == 0:
            return None
        if cache.key.shape[2] <= keep_tokens:
            return cache
        return _KVCache(
            key=cache.key[:, :, -keep_tokens:, :],
            value=cache.value[:, :, -keep_tokens:, :],
        )

    @staticmethod
    def _cache_length(caches: tuple[_KVCache | None, ...]) -> int:
        lengths = {cache.key.shape[2] for cache in caches if cache is not None}
        if not lengths:
            return 0
        if len(lengths) != 1:
            raise ValueError(f"Transformer video cache lengths differ: {lengths}.")
        return lengths.pop()


class FlashVSRConditionNet(nn.Module):
    """4x streaming VSR with rolling video KV and persistent Wan latents.

    The condition latent primes per-layer persistent KV. During training its
    final hidden states can additionally be decoded by a direct-RGB auxiliary
    head, while every LR frame uses a separate bicubic-residual output path.
    Inference still emits exactly one SR frame for every LR frame.
    """

    VARIANTS: ClassVar[dict[str, _VariantSpec]] = {
        "halfS": _VariantSpec(depth=7, dim=512, num_heads=8, num_kv_groups=2),
        "S": _VariantSpec(depth=10, dim=512, num_heads=8, num_kv_groups=2),
        "M": _VariantSpec(depth=11, dim=640, num_heads=10, num_kv_groups=2),
        "B": _VariantSpec(depth=12, dim=768, num_heads=12, num_kv_groups=3),
        "L": _VariantSpec(depth=14, dim=1024, num_heads=16, num_kv_groups=4),
        "XL": _VariantSpec(depth=14, dim=1536, num_heads=24, num_kv_groups=3),
    }

    def __init__(
        self,
        *,
        condition_channels: int = 48,
        condition_mode: FlashVSRConditionMode = "start",
        dim: int = 512,
        depth: int = 10,
        num_heads: int = 8,
        num_kv_groups: int = 2,
        encoder_channels: tuple[int, int, int] = (64, 128, 128),
        mlp_ratio: float = 8.0 / 3.0,
        spatial_head_dim: int = 1024,
        window_size: int = 3,
        upscale_factor: int = 4,
        rope_theta: float = 10_000.0,
        norm_eps: float = 1e-5,
        gradient_checkpointing: FlashVSRGradientCheckpointing = "none",
        spatial_attention: Mapping[str, Any] | None = None,
        training_execution: str = "streaming",
        temporal_rope_policy: str = "current_relative",
    ) -> None:
        super().__init__()

        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}.")
        if spatial_head_dim <= 0:
            raise ValueError(f"spatial_head_dim must be positive, got {spatial_head_dim}.")
        if gradient_checkpointing != "none":
            raise ValueError(
                "gradient_checkpointing must be one of 'none', 'mlp', or "
                f"'block', got {gradient_checkpointing!r}."
            )
        if condition_mode not in ("start", "endpoints"):
            raise ValueError(
                f"condition_mode must be 'start' or 'endpoints', got {condition_mode!r}."
            )
        hidden_dim = math.ceil(dim * mlp_ratio / 64) * 64
        self.spatial_attention = LocalAttentionConfig.from_mapping(spatial_attention)
        if training_execution != "streaming":
            raise ValueError("This distribution supports streaming inference only.")
        self.training_execution = training_execution
        if temporal_rope_policy != "current_relative":
            raise ValueError("FlashVSR temporal_rope_policy must be current_relative.")
        self.temporal_rope_policy = temporal_rope_policy
        if window_size < 1:
            raise ValueError("window_size must include at least the current frame.")
        self.condition_channels = condition_channels
        self.condition_mode = condition_mode
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.gradient_checkpointing = gradient_checkpointing
        self.upscale_factor = upscale_factor
        self.encoder_downsample_factor = 4
        self.pixel_shuffle_factor = self.encoder_downsample_factor * upscale_factor

        rope = _RotaryEmbedding3D(theta=rope_theta)
        head_dim = dim // num_heads
        if head_dim != rope.head_dim:
            raise ValueError(
                f"FlashVSRConditionNet requires {rope.head_dim}-dimensional "
                f"attention heads, got {head_dim}."
            )

        self.lr_encoder = _LRFrameEncoder(
            dim=dim,
            channels=encoder_channels,
            norm_eps=norm_eps,
        )
        self.condition_encoder = _ConditionEncoder(
            latent_channels=condition_channels,
            dim=dim,
            norm_eps=norm_eps,
        )
        self.video_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.condition_type_embedding = nn.Parameter(torch.zeros(1, 1, dim))
        self.backbone = _ConditionedStreamingTransformer(
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            num_kv_groups=num_kv_groups,
            hidden_dim=hidden_dim,
            max_video_frames=window_size,
            rope=rope,
            norm_eps=norm_eps,
            gradient_checkpointing=gradient_checkpointing,
            spatial_attention=self.spatial_attention,
        )
        self.spatial_head = nn.Sequential(
            nn.Linear(dim, spatial_head_dim),
            nn.SiLU(),
        )
        output_channels = 3 * self.pixel_shuffle_factor * self.pixel_shuffle_factor
        self.residual_projection = nn.Linear(spatial_head_dim, output_channels)
        self.condition_rgb_projection = nn.Linear(spatial_head_dim, output_channels)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    @classmethod
    def from_variant(
        cls,
        variant: FlashVSRConditionVariant = "S",
        **overrides: Any,
    ) -> FlashVSRConditionNet:
        normalized_variants = {name.upper(): spec for name, spec in cls.VARIANTS.items()}
        try:
            spec = normalized_variants[variant.upper()]
        except KeyError as error:
            supported = ", ".join(cls.VARIANTS)
            raise ValueError(
                f"Unknown FlashVSR condition variant {variant!r}; expected one of {supported}."
            ) from error
        config: dict[str, int | float | str | tuple[int, int, int]] = {
            "depth": spec.depth,
            "dim": spec.dim,
            "num_heads": spec.num_heads,
            "num_kv_groups": spec.num_kv_groups,
        }
        config.update(overrides)
        return cls(**config)

    def init_state(self) -> FlashVSRConditionState:
        return FlashVSRConditionState(
            video=(None,) * self.depth,
            condition=(None,) * self.depth,
        )

    @torch.no_grad()
    def prepare_condition(self, latent: torch.Tensor) -> FlashVSRPreparedCondition:
        """Encode one endpoint once, after weights are loaded and eval() is set."""
        if self.training:
            raise RuntimeError("Prepared conditions are only valid for inference.")
        if latent.ndim != 4 or latent.shape[1] != self.condition_channels:
            raise ValueError("Expected endpoint latent [B,condition_channels,H,W].")
        _, caches = self._prime_start_condition(
            latent, height=latent.shape[-2], width=latent.shape[-1]
        )
        return FlashVSRPreparedCondition(
            caches, latent.shape[0], latent.shape[-2], latent.shape[-1], id(self)
        )

    def begin_stream(
        self,
        start: FlashVSRPreparedCondition,
        end: FlashVSRPreparedCondition,
        *,
        total_frames: int,
        input_height: int,
        input_width: int,
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> FlashVSRConditionState:
        """Create a fresh LR history using two independently prepared endpoints."""
        if self.training or self.condition_mode != "endpoints":
            raise RuntimeError("begin_stream requires eval() and endpoint mode.")
        expected = ((input_height + 3) // 4, (input_width + 3) // 4)
        if total_frames < 2 or start.owner != id(self) or end.owner != id(self):
            raise ValueError("Invalid frame count or prepared condition owner.")
        if (start.height, start.width) != expected or (
            end.batch_size,
            end.height,
            end.width,
        ) != (start.batch_size, *expected):
            raise ValueError("Prepared endpoint shapes do not match the LR stream.")
        self._validate_end_condition_keep_mask(
            end_condition_keep_mask,
            batch_size=start.batch_size,
            device=start.caches[0].key.device,
        )
        caches = tuple(
            _KVCache(torch.cat((a.key, b.key), dim=2), torch.cat((a.value, b.value), dim=2))
            for a, b in zip(start.caches, end.caches, strict=True)
        )
        return FlashVSRConditionState(
            video=(None,) * self.depth,
            condition=caches,
            batch_size=start.batch_size,
            input_height=input_height,
            input_width=input_width,
            total_frames=total_frames,
            spatial_attention=self.spatial_attention,
            primed=True,
            end_condition_keep_mask=(
                end_condition_keep_mask.clone() if end_condition_keep_mask is not None else None
            ),
        )

    @torch.no_grad()
    def prepare_streaming(self, height: int, width: int, total_frames: int) -> None:
        """Precompute one relative RoPE plan per possible retained history length."""
        if self.training or total_frames < 2:
            raise ValueError("RoPE preparation requires eval() and at least 2 frames.")
        parameter = next(self.parameters())
        self.backbone._inference_rope_plans = {}
        h, w = (height + 3) // 4, (width + 3) // 4
        for index in range(min(total_frames, self.window_size)):
            self.backbone._build_video_rope(
                frame_index=index,
                previous_frames=min(index, self.window_size - 1),
                condition_temporal_positions=(0, total_frames - 1)
                if self.condition_mode == "endpoints"
                else (0,),
                height=h,
                width=w,
                device=parameter.device,
                dtype=parameter.dtype,
            )

    def decode_step(
        self,
        lr_frame: torch.Tensor,
        state: FlashVSRConditionState | None = None,
        *,
        condition_latent: torch.Tensor | None = None,
        end_condition_latent: torch.Tensor | None = None,
        total_frames: int | None = None,
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, FlashVSRConditionState]:
        output, next_state, _ = self._decode_step_impl(
            lr_frame,
            state,
            condition_latent=condition_latent,
            end_condition_latent=end_condition_latent,
            end_condition_keep_mask=end_condition_keep_mask,
            total_frames=total_frames,
            return_condition_hidden=False,
        )
        return output, next_state

    def _decode_step_impl(
        self,
        lr_frame: torch.Tensor,
        state: FlashVSRConditionState | None,
        *,
        condition_latent: torch.Tensor | None,
        end_condition_latent: torch.Tensor | None,
        total_frames: int | None,
        return_condition_hidden: bool,
        end_condition_keep_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        FlashVSRConditionState,
        torch.Tensor | None,
    ]:
        self._validate_lr_frame(lr_frame)
        batch_size, _, height, width = lr_frame.shape
        state = self.init_state() if state is None else state
        self._validate_state(
            state,
            batch_size=batch_size,
            height=height,
            width=width,
        )

        if end_condition_keep_mask is not None and (state.frame_index != 0 or state.primed):
            raise ValueError("end_condition_keep_mask is fixed at stream initialization.")
        state_end_condition_keep_mask = (
            end_condition_keep_mask.clone()
            if end_condition_keep_mask is not None
            else state.end_condition_keep_mask
        )
        self._validate_end_condition_keep_mask(
            state_end_condition_keep_mask, batch_size=batch_size, device=lr_frame.device
        )

        # Preserve the caller's size in stream state; pad only the model input.
        pad_h, pad_w = (-height) % 4, (-width) % 4
        if pad_h or pad_w:
            lr_frame = F.pad(lr_frame, (0, pad_w, 0, pad_h), mode="replicate")
        token_height = (height + pad_h) // self.encoder_downsample_factor
        token_width = (width + pad_w) // self.encoder_downsample_factor
        if state.frame_index == 0 and state.primed:
            if (
                condition_latent is not None
                or end_condition_latent is not None
                or total_frames is not None
            ):
                raise ValueError("Prepared conditions cannot be replaced during a stream.")
            condition_caches = state.condition
            condition_hidden = None
            state_total_frames = state.total_frames
            condition_temporal_positions = self._condition_temporal_positions(state)
        elif state.frame_index == 0:
            if condition_latent is None:
                raise ValueError("The first decode_step requires one first-frame condition latent.")
            self._validate_condition_latent(
                condition_latent,
                batch_size=batch_size,
                height=token_height,
                width=token_width,
            )
            if self.condition_mode == "start":
                if end_condition_latent is not None:
                    raise ValueError(
                        "end_condition_latent is only valid when condition_mode='endpoints'."
                    )
                if total_frames is not None:
                    raise ValueError("total_frames is only valid when condition_mode='endpoints'.")
                condition_hidden, condition_caches = self._prime_start_condition(
                    condition_latent,
                    height=token_height,
                    width=token_width,
                )
                condition_temporal_positions = (0,)
                state_total_frames = None
            else:
                if end_condition_latent is None:
                    raise ValueError(
                        "The first decode_step requires an end condition latent "
                        "when condition_mode='endpoints'."
                    )
                self._validate_condition_latent(
                    end_condition_latent,
                    batch_size=batch_size,
                    height=token_height,
                    width=token_width,
                    endpoint="end",
                )
                if total_frames is None or total_frames < 2:
                    raise ValueError("total_frames must be >= 2 when condition_mode='endpoints'.")
                condition_hidden, condition_caches = self._prime_endpoint_conditions(
                    condition_latent,
                    end_condition_latent,
                    batch_size=batch_size,
                    height=token_height,
                    width=token_width,
                )
                condition_temporal_positions = (0, total_frames - 1)
                state_total_frames = total_frames
        else:
            if condition_latent is not None or end_condition_latent is not None:
                raise ValueError(
                    "The condition latent is fixed at frame 0 and cannot be provided again."
                )
            if total_frames is not None:
                raise ValueError("total_frames can only be provided at frame 0.")
            condition_caches = state.condition
            condition_hidden = None
            state_total_frames = state.total_frames
            condition_temporal_positions = self._condition_temporal_positions(state)

        hidden_states = self.lr_encoder(lr_frame)
        hidden_states = hidden_states + self.video_type_embedding
        hidden_states, video_caches = self.backbone.forward_video(
            hidden_states,
            video_caches=state.video,
            condition_caches=condition_caches,
            frame_index=state.frame_index,
            condition_temporal_positions=condition_temporal_positions,
            end_condition_keep_mask=state_end_condition_keep_mask,
            height=token_height,
            width=token_width,
        )
        output = self._reconstruct(
            hidden_states,
            lr_frame=lr_frame,
            token_height=token_height,
            token_width=token_width,
        )
        output = output[..., : height * self.upscale_factor, : width * self.upscale_factor]
        next_state = FlashVSRConditionState(
            video=video_caches,
            condition=condition_caches,
            frame_index=state.frame_index + 1,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            total_frames=state_total_frames,
            spatial_attention=self.spatial_attention,
            end_condition_keep_mask=state_end_condition_keep_mask,
        )
        if not return_condition_hidden:
            condition_hidden = None
        return output, next_state, condition_hidden

    @torch.no_grad()
    def forward(self, lr_video, condition_latent, *, end_condition_latent=None):
        """Offline convenience wrapper over the identical streaming decoder."""
        if lr_video.ndim != 5 or lr_video.shape[1] < 1:
            raise ValueError("Expected non-empty LR video [B,T,3,H,W].")
        state, outputs = None, []
        for index, frame in enumerate(lr_video.unbind(1)):
            output, state = self.decode_step(
                frame,
                state,
                condition_latent=condition_latent if index == 0 else None,
                end_condition_latent=end_condition_latent if index == 0 else None,
                total_frames=lr_video.shape[1]
                if index == 0 and self.condition_mode == "endpoints"
                else None,
            )
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def _prime_start_condition(
        self,
        condition_latent: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, tuple[_KVCache, ...]]:
        condition_tokens = self.condition_encoder(condition_latent)
        condition_tokens = condition_tokens + self.condition_type_embedding
        return self.backbone.prime_condition(
            condition_tokens,
            height=height,
            width=width,
        )

    def _prime_endpoint_conditions(
        self,
        condition_latent: torch.Tensor,
        end_condition_latent: torch.Tensor,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, tuple[_KVCache, ...]]:
        endpoint_latents = torch.cat(
            (condition_latent, end_condition_latent),
            dim=0,
        )
        endpoint_tokens = self.condition_encoder(endpoint_latents)
        endpoint_tokens = endpoint_tokens + self.condition_type_embedding
        endpoint_hidden, endpoint_caches = self.backbone.prime_condition(
            endpoint_tokens,
            height=height,
            width=width,
        )
        endpoint_hidden = endpoint_hidden.unflatten(0, (2, batch_size)).transpose(0, 1)
        condition_caches = tuple(
            _KVCache(
                key=self._merge_endpoint_cache(cache.key, batch_size=batch_size),
                value=self._merge_endpoint_cache(cache.value, batch_size=batch_size),
            )
            for cache in endpoint_caches
        )
        return endpoint_hidden, condition_caches

    @staticmethod
    def _merge_endpoint_cache(
        tensor: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        tensor = tensor.unflatten(0, (2, batch_size)).permute(1, 2, 0, 3, 4)
        return tensor.flatten(2, 3)

    def _condition_temporal_positions(
        self,
        state: FlashVSRConditionState,
    ) -> tuple[int, ...]:
        if self.condition_mode == "start":
            return (0,)
        assert state.total_frames is not None
        return (0, state.total_frames - 1)

    def _reconstruct(
        self,
        hidden_states: torch.Tensor,
        *,
        lr_frame: torch.Tensor,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        batch_size = lr_frame.shape[0]
        head_features = self.spatial_head(hidden_states)
        residual = self._project_pixels(
            head_features,
            projection=self.residual_projection,
            batch_size=batch_size,
            token_height=token_height,
            token_width=token_width,
        )
        bicubic = F.interpolate(
            lr_frame,
            scale_factor=self.upscale_factor,
            mode="bicubic",
            align_corners=False,
        )
        return bicubic + residual

    def _project_pixels(
        self,
        hidden_states: torch.Tensor,
        *,
        projection: nn.Linear,
        batch_size: int,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        pixels = projection(hidden_states)
        pixels = pixels.transpose(1, 2).reshape(
            batch_size,
            3 * self.pixel_shuffle_factor * self.pixel_shuffle_factor,
            token_height,
            token_width,
        )
        return F.pixel_shuffle(pixels, self.pixel_shuffle_factor)

    def _validate_lr_frame(self, lr_frame: torch.Tensor) -> None:
        if lr_frame.ndim != 4:
            raise ValueError(
                f"FlashVSRConditionNet.decode_step expects [B,3,H,W], got {tuple(lr_frame.shape)}."
            )
        batch_size, channels, height, width = lr_frame.shape
        if batch_size <= 0 or height <= 0 or width <= 0:
            raise ValueError(f"LR frame dimensions must be positive, got {tuple(lr_frame.shape)}.")
        if channels != 3:
            raise ValueError(f"Expected 3 LR channels, got {channels}.")

    def _validate_end_condition_keep_mask(
        self,
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> None:
        if mask is None:
            return
        if self.condition_mode != "endpoints":
            raise ValueError("end_condition_keep_mask requires condition_mode='endpoints'.")
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.shape != (batch_size,)
        ):
            raise ValueError("end_condition_keep_mask must be boolean [B], True keeps the end.")
        if mask.device != device:
            raise ValueError("end_condition_keep_mask must be on the input device.")

    def _validate_condition_latent(
        self,
        condition_latent: torch.Tensor,
        *,
        batch_size: int,
        height: int,
        width: int,
        endpoint: str = "first-frame",
    ) -> None:
        expected = (batch_size, self.condition_channels, height, width)
        if tuple(condition_latent.shape) != expected:
            raise ValueError(
                f"Expected {endpoint} condition latent with shape "
                f"{expected}, got {tuple(condition_latent.shape)}."
            )

    def _validate_state(
        self,
        state: FlashVSRConditionState,
        *,
        batch_size: int,
        height: int,
        width: int,
    ) -> None:
        if len(state.video) != self.depth or len(state.condition) != self.depth:
            raise ValueError("State cache depth does not match the condition model depth.")
        if state.frame_index < 0:
            raise ValueError(f"State frame index must be non-negative: {state.frame_index}.")
        if state.frame_index == 0 and not state.primed:
            if state.end_condition_keep_mask is not None:
                raise ValueError("An empty state cannot contain an endpoint mask.")
            if any(cache is not None for cache in state.condition):
                raise ValueError("An empty state cannot contain condition caches.")
            if state.total_frames is not None:
                raise ValueError("An empty state cannot define total_frames.")
            return
        if state.spatial_attention != self.spatial_attention:
            raise ValueError("Spatial attention configuration changed; start a new stream.")
        if any(cache is None for cache in state.condition):
            raise ValueError("An active stream must contain all condition caches.")
        if self.condition_mode == "start":
            if state.total_frames is not None:
                raise ValueError("Start-condition state cannot define total_frames.")
        else:
            if state.total_frames is None or state.total_frames < 2:
                raise ValueError("Endpoint-condition state must define total_frames >= 2.")
            if state.frame_index >= state.total_frames:
                raise ValueError(
                    f"The endpoint-conditioned stream is complete at frame {state.total_frames}."
                )
        expected = (state.batch_size, state.input_height, state.input_width)
        actual = (batch_size, height, width)
        if expected != actual:
            raise ValueError(
                f"Streaming shape changed from {expected} to {actual}; start a "
                "new state for a different stream."
            )


__all__ = [
    "FlashVSRConditionMode",
    "FlashVSRConditionNet",
    "FlashVSRConditionState",
    "FlashVSRConditionVariant",
    "FlashVSRGradientCheckpointing",
    "FlashVSRPreparedCondition",
]
