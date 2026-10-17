"""Single-pass video decoding into an in-memory store of JPEG frames.

Every later stage (perception, VLM windows, evidence frames) samples from this store, so the
file is decoded exactly once.
"""
from __future__ import annotations

import bisect
import io
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np
from PIL import Image


@dataclass
class FrameStore:
    fps: float
    duration: float
    width: int
    height: int
    times: list[float] = field(default_factory=list)
    jpegs: list[bytes] = field(default_factory=list)

    def index_at(self, t: float) -> int:
        """Index of the stored frame nearest to time t."""
        i = bisect.bisect_left(self.times, t)
        if i <= 0:
            return 0
        if i >= len(self.times):
            return len(self.times) - 1
        return i if self.times[i] - t < t - self.times[i - 1] else i - 1

    def image(self, i: int) -> Image.Image:
        return Image.open(io.BytesIO(self.jpegs[i])).convert("RGB")

    def sample(self, start: float, end: float, fps: float) -> list[int]:
        """Indices of frames between start and end at roughly `fps`, unique and ordered."""
        start = max(0.0, start)
        end = min(self.duration, end)
        if end <= start:
            return [self.index_at(start)]
        n = max(1, int(round((end - start) * fps)))
        step = (end - start) / n
        idx = [self.index_at(start + step * (k + 0.5)) for k in range(n)]
        return sorted(set(idx))


def _rotation_k(frame: av.VideoFrame) -> int:
    """np.rot90 k needed to display the frame upright (phone videos carry a display matrix)."""
    rot = getattr(frame, "rotation", 0) or 0
    theta = int(round(-rot)) % 360
    return {90: -1, 180: 2, 270: 1}.get(theta, 0)


def decode(path: str | Path, fps: float = 5.0, quality: int = 88) -> FrameStore:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / av.time_base
        else:
            duration = 0.0

        store: FrameStore | None = None
        next_t = 0.0
        last_t = 0.0
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if stream.start_time is not None:
                t -= float(stream.start_time * stream.time_base)
            last_t = t
            if t + 1e-6 < next_t:
                continue
            img = frame.to_ndarray(format="rgb24")
            k = _rotation_k(frame)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
            if store is None:
                store = FrameStore(fps=fps, duration=duration, width=img.shape[1], height=img.shape[0])
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            store.times.append(round(t, 3))
            store.jpegs.append(buf.getvalue())
            next_t += 1.0 / fps
            while next_t <= t:
                next_t += 1.0 / fps

    if store is None:
        raise ValueError("no decodable video frames")
    store.duration = max(store.duration, last_t)
    return store


def resize_for_vlm(img: Image.Image, width: int) -> Image.Image:
    """Resize so both sides are multiples of 32 (Qwen-VL merged patch size)."""
    w, h = img.size
    nw = max(32, int(round(width / 32)) * 32)
    nh = max(32, int(round(h * nw / w / 32)) * 32)
    return img.resize((nw, nh), Image.BICUBIC)


def to_jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
