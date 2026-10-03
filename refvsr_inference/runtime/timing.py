"""Nonblocking device spans, resolved only after output completion."""

import time

import torch


class DeviceSpan:
    def __init__(self, device):
        self.cuda = torch.device(device).type == "cuda"
        self.started = time.perf_counter()
        self.cpu_ms = None
        if self.cuda:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()

    def finish(self):
        if self.cuda:
            self.end.record()
        self.cpu_ms = (time.perf_counter() - self.started) * 1000
        return self

    def milliseconds(self):
        if self.cpu_ms is None:
            raise RuntimeError("Unfinished device span.")
        if self.cuda:
            self.end.synchronize()
            return self.start.elapsed_time(self.end)
        return self.cpu_ms
