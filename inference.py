# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Run with `python inference.py` after installing the project environment."""

import argparse
import time
from pathlib import Path

import torch

from refvsr_inference.checkpoints import load_checkpoints
from refvsr_inference.media import VideoWriter, input_fps, iter_frames
from refvsr_inference.models.wan.sampling import WanVSRPipeline
from refvsr_inference.runtime.streaming import CollaborativeFrameStream, InputFrame


def parse_args():
    parser = argparse.ArgumentParser(description="RefVSR 4x video super-resolution")
    project_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--input", type=Path, default=project_root / "assets" / "demo" / "input.mp4",
        help="Video or ordered frame directory (default: assets/demo/input.mp4)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output MP4")
    weights = project_root / "weights"
    parser.add_argument(
        "--wan-checkpoint", type=Path, default=weights / "wan.safetensors",
        help="Wan checkpoint (default: weights/wan.safetensors)",
    )
    parser.add_argument(
        "--flash-checkpoint", type=Path, default=weights / "flash.safetensors",
        help="Flash checkpoint (default: weights/flash.safetensors, S step-60000 EMA)",
    )
    parser.add_argument("--window-size", type=int, default=16, help="Reference interval + 1")
    parser.add_argument("--fps", help="Override FPS; required for frame directories")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main(args):
    if args.window_size < 2 or args.cpu_threads < 1:
        raise ValueError("Require window-size >= 2 and cpu-threads >= 1.")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Output must end with .mp4.")
    if args.output.resolve() == args.input.resolve():
        raise ValueError("Input and output must differ.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite to replace it.")
    for checkpoint in (args.wan_checkpoint, args.flash_checkpoint):
        if checkpoint.suffix != ".safetensors" or not checkpoint.is_file():
            raise FileNotFoundError(f"RefVSR checkpoint not found: {checkpoint}")
    torch.set_num_threads(args.cpu_threads)
    fps = input_fps(args.input, args.fps)
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    wan, flash = load_checkpoints(
        args.wan_checkpoint, args.flash_checkpoint, device=device, dtype=dtype
    )
    sampler = WanVSRPipeline(
        scheduler_config={"num_train_timesteps": 1000, "shift": 5.0},
        sampling={
            "execution_mode": "streaming",
            "num_inference_steps": 1,
            "timesteps": [1000],
            "guidance_scale": 1.0,
            "seed": args.seed,
            "kv_cache_window_size": wan.temporal_context_frames,
        },
    )
    sampler.prepare_streaming(
        model=wan, prompt_embeds=wan.prompt_embeds, device=device, compute_dtype=dtype
    )
    stream = CollaborativeFrameStream(
        wan_model=wan,
        flash_model=flash,
        wan_pipeline=sampler,
        prompt_embeds=wan.prompt_embeds,
        device=device,
        compute_dtype=dtype,
        window_size=args.window_size,
        seed=args.seed,
    )
    frames = iter_frames(args.input, max_frames=args.max_frames, start_frame=args.start_frame)
    packets = (InputFrame(i, rgb, source_time=i / float(fps)) for i, rgb in enumerate(frames))
    started = time.perf_counter()
    try:
        with VideoWriter(args.output, fps) as writer:
            for result in stream.frames(packets):
                rgb = result.rgb[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8)
                writer.write(rgb.permute(1, 2, 0).contiguous().cpu())
    finally:
        packets.close()
        frames.close()
    print(f"Saved {writer.frames} frames to {args.output} ({time.perf_counter() - started:.2f}s)")


if __name__ == "__main__":
    main(parse_args())
