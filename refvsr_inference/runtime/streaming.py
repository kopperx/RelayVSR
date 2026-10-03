"""Bounded-lookahead collaboration with immediate, ordered frame delivery."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import islice

import torch
import torch.nn.functional as F

from .timing import DeviceSpan


@dataclass(frozen=True)
class InputFrame:
    index: int
    rgb: torch.Tensor  # uint8 CHW, native LR
    source_time: float = 0.0  # seconds relative to the session origin
    arrival_time: float = 0.0


@dataclass(frozen=True)
class OutputFrame:
    index: int
    rgb: torch.Tensor  # GPU/CPU float BCHW in [-1,1]
    source_time: float
    arrival_time: float
    window_start: int
    window_end: int
    timings: tuple[tuple[str, DeviceSpan], ...] = ()


def prepare_lr(frame: torch.Tensor, *, device, dtype, stride: int = 8):
    """Normalize native RGB and pad for the Wan latent/patch grid."""
    if frame.ndim != 3 or frame.shape[0] != 3 or frame.dtype != torch.uint8:
        raise ValueError("Streaming input must be uint8 RGB [3,H,W].")
    h, w = frame.shape[-2:]
    if min(h, w) <= 0:
        raise ValueError("Frame dimensions must be positive.")
    x = frame.unsqueeze(0).to(device=device, non_blocking=True).to(dtype)
    x = x.div(127.5).sub(1.0)
    return F.pad(x, (0, (-w) % stride, 0, (-h) % stride), mode="replicate")


class CollaborativeFrameStream:
    """Model-only generator shared by deployment and paced benchmarking.

    It buffers at most K native frames, retains four (configured) Wan keyframes,
    and resets Flash video history at each overlapping endpoint window. Yielding
    suspends before the next Flash decode, so downstream sees individual frames.
    """

    def __init__(
        self,
        *,
        wan_model,
        flash_model,
        wan_pipeline,
        prompt_embeds,
        device,
        compute_dtype,
        window_size=16,
        seed=42,
        reuse_endpoint_kv=True,
        precompute_rope=True,
    ):
        if window_size < 2:
            raise ValueError("Collaborative window_size must be at least 2.")
        if flash_model.condition_mode != "endpoints":
            raise ValueError("Collaborative streaming requires endpoint conditions.")
        if flash_model.training or wan_model.training:
            raise ValueError("Streaming models must be in eval() mode.")
        self.wan = wan_model
        self.flash = flash_model
        self.pipeline = wan_pipeline
        self.prompt = prompt_embeds
        self.device = torch.device(device)
        self.dtype = compute_dtype
        self.window_size = int(window_size)
        self.seed = seed
        self.reuse_endpoint_kv = reuse_endpoint_kv
        self.precompute_rope = precompute_rope
        self.last_keyframe_indices: deque[int] = deque(maxlen=10000)
        self.keyframe_count = 0
        self.capture_state = False

    @torch.no_grad()
    def prepare(self, height, width):
        patch = self.wan.config.patch_size
        stride = 16 * max(int(patch[1]), int(patch[2])) // 4
        geometry = (
            height + (-height) % stride,
            width + (-width) % stride,
            self.window_size,
            self.dtype,
            self.device,
        )
        if self.precompute_rope and getattr(self, "_prepared_geometry", None) != geometry:
            self.flash.prepare_streaming(geometry[0], geometry[1], self.window_size)
            self._prepared_geometry = geometry

    @torch.no_grad()
    def frames(self, source: Iterable[InputFrame]) -> Iterator[OutputFrame]:
        iterator = iter(source)
        first = next(iterator, None)
        if first is None:
            raise ValueError("Cannot infer an empty stream.")
        if first.index != 0:
            raise ValueError("A new stream must start at frame 0.")
        shape = tuple(first.rgb.shape)
        h, w = shape[-2:]
        patch = self.wan.config.patch_size
        stride = 16 * max(int(patch[1]), int(patch[2])) // 4
        ph, pw = h + (-h) % stride, w + (-w) % stride
        self.prepare(h, w)
        wan_state = self.pipeline.create_streaming_state(self.wan)
        generator = (
            None
            if self.seed is None
            else [torch.Generator(device=self.device).manual_seed(self.seed)]
        )
        self.last_keyframe_indices.clear()
        self.keyframe_count = 0

        pending_timings = []

        def keyframe(packet):
            span = DeviceSpan(self.device)
            native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
            upsampled = F.interpolate(native, scale_factor=4, mode="bilinear", align_corners=False)
            pending_timings.append(("wan_input", span.finish()))
            span = DeviceSpan(self.device)
            latent = self.pipeline.sample_streaming_latent_step(
                model=self.wan,
                lq_frame=upsampled.unsqueeze(2),
                spatial_scale_factor=16,
                prompt_embeds=self.prompt,
                streaming_state=wan_state,
                generator=generator,
                device=self.device,
                compute_dtype=self.dtype,
            )[:, :, 0].to(self.dtype)
            pending_timings.append(("wan", span.finish()))
            self.last_keyframe_indices.append(packet.index)
            self.keyframe_count += 1
            if self.capture_state:
                self.last_wan_latent = latent
            span = DeviceSpan(self.device)
            prepared = self.flash.prepare_condition(latent) if self.reuse_endpoint_kv else None
            pending_timings.append(("flash_condition", span.finish()))
            return latent, prepared

        # First condition can be computed while the producer collects lookahead.
        start_latent, start_prepared = keyframe(first)
        start = first
        first_window = True
        while True:
            window = [start, *islice(iterator, self.window_size - 1)]
            for offset, packet in enumerate(window):
                if packet.index != start.index + offset or tuple(packet.rgb.shape) != shape:
                    raise ValueError(
                        "Streaming frames must be consecutive with constant dimensions."
                    )
            if len(window) == 1 and not first_window:
                return
            end = window[-1]
            if end.index == start.index:
                end_latent, end_prepared = start_latent, start_prepared
            else:
                end_latent, end_prepared = keyframe(end)
            count = max(2, len(window))
            span = DeviceSpan(self.device)
            state = (
                self.flash.begin_stream(
                    start_prepared,
                    end_prepared,
                    total_frames=count,
                    input_height=ph,
                    input_width=pw,
                )
                if self.reuse_endpoint_kv
                else None
            )
            pending_timings.append(("flash_condition", span.finish()))
            for offset, packet in enumerate(window):
                span = DeviceSpan(self.device)
                native = prepare_lr(packet.rgb, device=self.device, dtype=self.dtype, stride=stride)
                kwargs = {}
                if offset == 0 and not self.reuse_endpoint_kv:
                    kwargs = {
                        "condition_latent": start_latent,
                        "end_condition_latent": end_latent,
                        "total_frames": count,
                    }
                pending_timings.append(("flash_input", span.finish()))
                span = DeviceSpan(self.device)
                prediction, state = self.flash.decode_step(native, state, **kwargs)
                pending_timings.append(("flash", span.finish()))
                if self.capture_state:
                    self.last_flash_state = state
                # The overlap must update history, but has already been emitted.
                if first_window or offset:
                    timings = tuple(pending_timings)
                    pending_timings.clear()
                    yield OutputFrame(
                        packet.index,
                        prediction[..., : h * 4, : w * 4],
                        packet.source_time,
                        packet.arrival_time,
                        start.index,
                        end.index,
                        timings,
                    )
            first_window = False
            start, start_latent, start_prepared = end, end_latent, end_prepared
            if len(window) < self.window_size:
                return
