"""Inference-only flow-matching scheduler helpers."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def normalize_flow_match_scheduler_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the pretrained Wan UniPC config to FlowMatch Euler fields."""
    shift = config.get("shift", config.get("flow_shift"))
    if shift is None:
        raise ValueError("Wan scheduler config must define shift or flow_shift.")
    return {
        "num_train_timesteps": int(config["num_train_timesteps"]),
        "shift": float(shift),
    }


def set_custom_flow_timesteps(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timesteps: Sequence[float],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Set exact model timesteps without applying the scheduler shift twice."""
    values = torch.as_tensor(timesteps, dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Custom flow timesteps must be a non-empty sequence.")
    if not torch.isfinite(values).all():
        raise ValueError("Custom flow timesteps must contain only finite values.")

    num_train_timesteps = float(scheduler.config.num_train_timesteps)
    if ((values <= 0) | (values > num_train_timesteps)).any():
        raise ValueError(f"Custom flow timesteps must be within (0, {int(num_train_timesteps)}].")
    if values.numel() > 1 and not torch.all(values[:-1] > values[1:]):
        raise ValueError("Custom flow timesteps must be strictly descending.")

    shift = float(scheduler.config.shift)
    if shift <= 0:
        raise ValueError("Flow-matching scheduler shift must be positive.")

    target_sigmas = values / num_train_timesteps
    raw_sigmas = target_sigmas / (shift - (shift - 1.0) * target_sigmas)
    scheduler.set_timesteps(
        sigmas=raw_sigmas.tolist(),
        device=device,
    )
    return scheduler.timesteps


def expand_timesteps_to_token_sequence(
    model: torch.nn.Module,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    patch_size = _get_patch_size(model)
    _, _, num_frames, height, width = latents.shape
    p_t, p_h, p_w = patch_size
    if num_frames % p_t != 0 or height % p_h != 0 or width % p_w != 0:
        raise ValueError(
            "Latent shape is not divisible by Wan patch size: "
            f"latents={tuple(latents.shape)}, patch_size={patch_size}."
        )

    token_count = (num_frames // p_t) * (height // p_h) * (width // p_w)
    return timesteps[:, None].expand(-1, token_count)


def _get_patch_size(model: torch.nn.Module) -> tuple[int, int, int]:
    patch_size = model.config.patch_size
    if len(patch_size) != 3:
        raise ValueError(f"Expected 3D patch_size, got {patch_size!r}.")
    return tuple(int(value) for value in patch_size)
