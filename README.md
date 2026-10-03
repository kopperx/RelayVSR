# RelayVSR

Collaborative streaming 4× video super-resolution with sparse Wan references
and a compact conditional decoder. This repository contains inference code only.

## Setup

Linux, Python 3.11–3.13 and an NVIDIA GPU:

```bash
uv sync --locked --extra cu128
source .venv/bin/activate
```

## Inference

Download the two checkpoints from the anonymous
[Hugging Face repository](https://huggingface.co/anonyaa/RelayVSR) into `weights/`:

```bash
uvx hf download anonyaa/RelayVSR flash.safetensors wan.safetensors --local-dir weights
```

The expected paths are `weights/wan.safetensors` and
`weights/flash.safetensors`, resolved relative to `inference.py`.
The bundled `assets/demo/input.mp4` is used when `--input` is omitted.

```bash
python inference.py \
  --output output_x4.mp4
```

Pass `--input path/to/video.mp4` to process another video, or provide an
ordered image directory together with `--fps`.

The default Flash checkpoint is **S, step 60000, EMA**, trained with a two-frame
video attention window (current frame and one predecessor) and current-relative
temporal RoPE. Override either path with `--wan-checkpoint` or `--flash-checkpoint`.

- `--window-size 16`: generate reference endpoints every 15 frames. Increase
  this value to let the small model process more intermediate frames.
- `--fps 24`: override the input FPS; required for image directories.
- `--max-frames 100`: process at most 100 frames.
- `--dtype bf16`: inference precision; `fp16` and `fp32` are also supported.
- `--overwrite`: replace an existing output.

Video FPS is preserved. Output resolution is exactly 4× the input; audio is not
copied. Each model loads its own safetensors directly, without runtime adapter
merging. Both files contain their model configuration; the Wan file also contains
the prompt embeddings.

## License

Apache-2.0. The Wan implementation retains its upstream copyright notice.
Acknowledgments: [Diffusers](https://github.com/huggingface/diffusers),
[Wan](https://github.com/Wan-Video/Wan2.2) and
[FlashVSR](https://github.com/OpenImagingLab/FlashVSR).
