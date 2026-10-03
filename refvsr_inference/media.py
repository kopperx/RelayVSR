"""Incremental RGB input and atomic MP4 output, with rational frame rates."""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def input_fps(path, override=None):
    if override is not None:
        fps = Fraction(str(override))
    else:
        path = Path(path)
        if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES:
            raise ValueError("Image inputs require --fps, for example --fps 24.")
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("Input has no video stream.")
            fps = container.streams.video[0].average_rate
            if fps is None:
                raise ValueError("Video frame rate is unavailable; specify --fps.")
    if fps <= 0:
        raise ValueError("FPS must be positive.")
    return fps


def iter_frames(path, *, max_frames=None, start_frame=0):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if start_frame < 0 or (max_frames is not None and max_frames < 1):
        raise ValueError("Require start_frame >= 0 and max_frames >= 1.")

    def images():
        paths = (
            sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
            if path.is_dir()
            else [path]
        )
        for image_path in paths:
            with Image.open(image_path) as image:
                yield np.asarray(image.convert("RGB")).copy()

    def video():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for frame in container.decode(stream):
                yield frame.to_ndarray(format="rgb24")

    source = images() if path.is_dir() or path.suffix.lower() in IMAGE_SUFFIXES else video()
    count = 0
    try:
        for index, rgb in enumerate(source):
            if index < start_frame:
                continue
            yield torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).contiguous()
            count += 1
            if count == max_frames:
                break
    finally:
        source.close()
    if count == 0:
        raise ValueError("No frames were decoded from the requested input range.")


class VideoWriter:
    def __init__(self, path, fps):
        self.path, self.fps = Path(path), fps
        self.container = self.stream = None
        self.frames = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=".refvsr-", suffix=".mp4", dir=self.path.parent, delete=False
        )
        self.temporary = Path(handle.name)
        handle.close()
        self.container = av.open(str(self.temporary), mode="w", format="mp4")
        return self

    def write(self, frame):
        rgb = frame.numpy() if isinstance(frame, torch.Tensor) else frame
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB [H,W,3].")
        h, w = rgb.shape[:2]
        if self.stream is None:
            self.stream = self.container.add_stream("libx264", rate=self.fps)
            self.stream.height, self.stream.width = h, w
            self.stream.pix_fmt = "yuv420p" if h % 2 == w % 2 == 0 else "yuv444p"
            self.stream.options = {"crf": "18", "preset": "fast"}
            self.stream.codec_context.thread_count = 1
        if (h, w) != (self.stream.height, self.stream.width):
            raise ValueError("Output dimensions changed.")
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
            self.container.mux(packet)
        self.frames += 1

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                if self.stream is None:
                    raise ValueError("Cannot publish an empty video.")
                for packet in self.stream.encode():
                    self.container.mux(packet)
                self.container.close()
                self.container = None
                self.temporary.replace(self.path)
        finally:
            if self.container is not None:
                self.container.close()
            self.temporary.unlink(missing_ok=True)
