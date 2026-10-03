"""Load separate Wan and Flash safetensors checkpoints."""

import json

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import load_file

from .models.flash.model import FlashVSRConditionNet
from .models.wan.transformer import WanTransformer3DModel


@torch.no_grad()
def load_checkpoints(wan_path, flash_path, *, device="cuda", dtype=torch.bfloat16):
    """Wan includes its prompt; both files include their architecture metadata."""
    with safe_open(str(wan_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "refvsr-wan-v1":
            raise ValueError("Expected a RefVSR Wan safetensors checkpoint.")
        wan_config = json.loads(metadata["config"])
        prompt_shape = handle.get_slice("prompt_embeds").get_shape()
        if (
            len(prompt_shape) != 3
            or prompt_shape[0] != 1
            or prompt_shape[1] < 1
            or prompt_shape[2] != wan_config["text_dim"]
        ):
            raise ValueError("Checkpoint prompt shape does not match the Wan model.")
    with safe_open(str(flash_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        if metadata.get("format") != "refvsr-flash-v1":
            raise ValueError("Expected a RefVSR Flash safetensors checkpoint.")
        flash_config = json.loads(metadata["config"])
    if (
        flash_config.get("condition_mode") != "endpoints"
        or flash_config.get("temporal_rope_policy") != "current_relative"
    ):
        raise ValueError("Flash checkpoint must use endpoints and current_relative RoPE.")
    if flash_config.get("upscale_factor", 4) != 4:
        raise ValueError("The collaborative model requires 4x upscaling.")
    variant = flash_config.pop("variant", "S")
    with init_empty_weights():
        wan = WanTransformer3DModel.from_config(wan_config)
        wan.lq_proj = wan.build_lq_proj()
        wan.register_buffer("prompt_embeds", torch.empty(prompt_shape))
        flash = FlashVSRConditionNet.from_variant(variant, **flash_config)
    wan.load_state_dict(load_file(str(wan_path), device="cpu"), strict=True, assign=True)
    flash.load_state_dict(load_file(str(flash_path), device="cpu"), strict=True, assign=True)
    wan = wan.to(device=device, dtype=dtype).eval().requires_grad_(False)
    flash = flash.to(device=device, dtype=dtype).eval().requires_grad_(False)
    return wan, flash
