"""Single-step Wan latent sampling used by collaborative inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from .flow import expand_timesteps_to_token_sequence, set_custom_flow_timesteps


@dataclass(frozen=True, slots=True)
class WanVSRSamplingConfig:
    execution_mode: Literal["causal", "streaming"] = "causal"
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    seed: int | None = None
    timesteps: list[float] | None = None
    shift: float | None = None
    kv_cache_window_size: int = -1
    fps: int = 8

    def __post_init__(self) -> None:
        if self.guidance_scale < 1.0:
            raise ValueError("guidance_scale must be greater than or equal to 1.")
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive.")
        if self.shift is not None and self.shift <= 0:
            raise ValueError("sampling.shift must be positive.")
        if self.execution_mode not in {"causal", "streaming"}:
            raise ValueError(
                "execution_mode must be either 'causal' or 'streaming', got "
                f"{self.execution_mode!r}."
            )
        if self.execution_mode == "streaming" and self.guidance_scale != 1.0:
            raise ValueError(
                "streaming execution_mode requires guidance_scale=1; "
                "CFG is only supported by causal execution_mode."
            )
        if self.execution_mode == "streaming" and (
            self.num_inference_steps != 1
            or (self.timesteps is not None and len(self.timesteps) != 1)
        ):
            raise ValueError("streaming requires exactly one inference step and one timestep.")
        if self.kv_cache_window_size == 0 or self.kv_cache_window_size < -1:
            raise ValueError("kv_cache_window_size must be -1 or a positive integer.")
        if self.execution_mode != "streaming" and self.kv_cache_window_size != -1:
            raise ValueError("kv_cache_window_size is only supported by streaming execution_mode.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> WanVSRSamplingConfig:
        return cls(
            execution_mode=str(config.get("execution_mode", "causal")).lower(),
            num_inference_steps=int(config.get("num_inference_steps", 50)),
            guidance_scale=float(config.get("guidance_scale", 5.0)),
            seed=(int(config["seed"]) if config.get("seed") is not None else None),
            timesteps=(
                [float(timestep) for timestep in config["timesteps"]]
                if config.get("timesteps") is not None
                else None
            ),
            shift=(float(config["shift"]) if config.get("shift") is not None else None),
            kv_cache_window_size=int(config.get("kv_cache_window_size", -1)),
            fps=int(config.get("fps", 8)),
        )


@dataclass(frozen=True, slots=True)
class WanVSRStreamingState:
    """One model state for a single-step streaming video batch."""

    model_state: Any


class WanVSRPipeline:
    """Prepare and sample normalized Wan keyframe latents."""

    def __init__(
        self,
        *,
        scheduler_config: Mapping[str, Any],
        sampling: Mapping[str, Any] | WanVSRSamplingConfig,
    ) -> None:
        self.sampling = (
            sampling
            if isinstance(sampling, WanVSRSamplingConfig)
            else WanVSRSamplingConfig.from_mapping(sampling)
        )
        resolved_scheduler_config = dict(scheduler_config)
        if self.sampling.timesteps is None and self.sampling.shift is not None:
            resolved_scheduler_config["shift"] = self.sampling.shift
        self._scheduler = FlowMatchEulerDiscreteScheduler.from_config(resolved_scheduler_config)
        self._streaming_conditioning_cache: Any | None = None

    @torch.no_grad()
    def prepare_streaming(
        self,
        *,
        model: torch.nn.Module,
        prompt_embeds: torch.Tensor,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> None:
        """Prepare static streaming conditioning before sampling any videos."""
        if prompt_embeds.shape[0] != 1:
            raise ValueError("Streaming prepare requires prompt batch size 1.")

        timesteps = self._set_timesteps(device)
        patch_size = tuple(int(value) for value in model.config.patch_size)
        _, patch_height, patch_width = patch_size
        dummy_latents = torch.empty(
            (
                1,
                int(model.config.in_channels),
                1,
                patch_height,
                patch_width,
            ),
            device=device,
            dtype=compute_dtype,
        )
        if len(timesteps) != 1:
            raise ValueError("streaming requires exactly one resolved scheduler timestep.")
        timestep = timesteps[0]
        timestep_batch = timestep.detach().to(device=device, dtype=torch.float32).reshape(1)
        timestep_tokens = expand_timesteps_to_token_sequence(
            model,
            dummy_latents,
            timestep_batch,
        )
        preparation_state = model.create_streaming_state()
        model(
            lq_videos=None,
            hidden_states=dummy_latents,
            timestep=timestep_tokens,
            encoder_hidden_states=prompt_embeds,
            streaming_state=preparation_state,
            use_lq_condition=False,
            precompute_streaming=True,
        )
        conditioning_cache = getattr(
            preparation_state,
            "conditioning_cache",
            None,
        )
        if conditioning_cache is None:
            raise RuntimeError("Streaming model did not populate its conditioning cache.")
        self._streaming_conditioning_cache = conditioning_cache

    def create_streaming_state(
        self,
        model: torch.nn.Module,
    ) -> WanVSRStreamingState:
        """Create one bounded Wan state per configured sampling timestep."""
        if self.sampling.execution_mode != "streaming":
            raise RuntimeError(
                "Latent-only streaming requires sampling.execution_mode='streaming'."
            )
        if self._streaming_conditioning_cache is None:
            raise RuntimeError(
                "Streaming conditioning was not prepared; call "
                "pipeline.prepare_streaming() before sampling."
            )
        return WanVSRStreamingState(
            model_state=model.create_streaming_state(
                kv_cache_window_size=self.sampling.kv_cache_window_size,
                conditioning_cache=self._streaming_conditioning_cache,
            )
        )

    @torch.no_grad()
    def sample_streaming_latent_step(
        self,
        *,
        model: torch.nn.Module,
        lq_frame: torch.Tensor,
        spatial_scale_factor: int,
        prompt_embeds: torch.Tensor,
        streaming_state: WanVSRStreamingState,
        generator: list[torch.Generator] | None,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Generate one normalized Wan latent without decoding it to RGB."""
        if self.sampling.execution_mode != "streaming":
            raise RuntimeError(
                "Latent-only streaming requires sampling.execution_mode='streaming'."
            )
        if lq_frame.ndim != 5 or lq_frame.shape[2] != 1:
            raise ValueError(
                f"A Wan streaming latent step expects [B,C,1,H,W], got {tuple(lq_frame.shape)}."
            )
        if prompt_embeds.shape[0] != 1:
            raise ValueError("Wan streaming prompt embeddings must have batch size 1.")
        if spatial_scale_factor <= 0:
            raise ValueError("spatial_scale_factor must be positive.")
        if (
            lq_frame.shape[-2] % spatial_scale_factor != 0
            or lq_frame.shape[-1] % spatial_scale_factor != 0
        ):
            raise ValueError(
                "Wan LQ height and width must be divisible by "
                f"spatial_scale_factor={spatial_scale_factor}, got "
                f"{tuple(lq_frame.shape[-2:])}."
            )

        frame_latents = _prepare_wan_latents(
            model=model,
            spatial_scale_factor=spatial_scale_factor,
            batch_size=int(lq_frame.shape[0]),
            height=int(lq_frame.shape[-2]),
            width=int(lq_frame.shape[-1]),
            num_frames=1,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        return self._sample_streaming_frame(
            model=model,
            frame_lq_videos=lq_frame,
            frame_latents=frame_latents,
            positive_prompt_embeds=prompt_embeds,
            streaming_state=streaming_state,
            device=device,
            compute_dtype=compute_dtype,
        )

    def _sample_streaming_frame(
        self,
        *,
        model: torch.nn.Module,
        frame_lq_videos: torch.Tensor,
        frame_latents: torch.Tensor,
        positive_prompt_embeds: torch.Tensor,
        streaming_state: WanVSRStreamingState,
        device: torch.device,
        compute_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps = self._set_timesteps(device)
        if len(timesteps) != 1:
            raise ValueError("streaming requires exactly one resolved scheduler timestep.")
        timestep = timesteps[0]
        timestep_batch = (
            timestep.detach()
            .to(device=device, dtype=torch.float32)
            .reshape(1)
            .expand(frame_latents.shape[0])
        )
        timestep_tokens = expand_timesteps_to_token_sequence(
            model,
            frame_latents,
            timestep_batch,
        )
        model_pred = model(
            lq_videos=frame_lq_videos,
            hidden_states=frame_latents.to(dtype=compute_dtype),
            timestep=timestep_tokens,
            encoder_hidden_states=positive_prompt_embeds.expand(
                frame_latents.shape[0],
                -1,
                -1,
            ),
            streaming_state=streaming_state.model_state,
        )
        frame_latents = self._scheduler.step(
            model_pred.float(),
            timestep,
            frame_latents,
            return_dict=False,
        )[0]
        return frame_latents

    def _set_timesteps(self, device: torch.device) -> torch.Tensor:
        if self.sampling.timesteps is not None:
            return set_custom_flow_timesteps(
                self._scheduler,
                self.sampling.timesteps,
                device=device,
            )
        self._scheduler.set_timesteps(
            num_inference_steps=self.sampling.num_inference_steps,
            device=device,
        )
        return self._scheduler.timesteps


def _prepare_wan_latents(
    *,
    model: torch.nn.Module,
    spatial_scale_factor: int,
    batch_size: int,
    height: int,
    width: int,
    num_frames: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: list[torch.Generator] | None,
) -> torch.Tensor:
    shape = (
        batch_size,
        int(model.config.in_channels),
        num_frames,
        height // spatial_scale_factor,
        width // spatial_scale_factor,
    )
    latents = torch.empty(shape, device=device, dtype=dtype)
    frame_shape = (shape[0], shape[1], 1, shape[3], shape[4])
    for frame in range(num_frames):
        latents[:, :, frame : frame + 1] = randn_tensor(
            frame_shape, generator=generator, device=device, dtype=dtype
        )
    return latents
