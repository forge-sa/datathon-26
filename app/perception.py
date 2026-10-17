"""Person detection (RT-DETR, Apache 2.0) and simple scene signals.

No faces, no identities: only person boxes per frame. Roles come from position: the cashier
zone is the area where a person stays the longest (estimated from the video itself, so it
adapts to a new camera angle).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from .video import FrameStore

_lock = threading.Lock()   # one user at a time; model loaded once
_model = None
_processor = None
_device = "cpu"


def _load(model_name: str, device: str = "cpu"):
    """device "cpu" (default) keeps the whole 12 GB GPU for the VLM: on Windows an over-full GPU
    silently spills into system RAM and every VLM call gets 10-30x slower."""
    global _model, _processor, _device
    if _model is not None:
        return
    import torch
    from transformers import AutoImageProcessor, RTDetrForObjectDetection

    _device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    _processor = AutoImageProcessor.from_pretrained(model_name)
    _model = RTDetrForObjectDetection.from_pretrained(model_name).to(_device).eval()
    if _device == "cuda":
        _model = _model.half()


@dataclass
class PersonTrack:
    times: list[float] = field(default_factory=list)
    boxes: list[list[list[float]]] = field(default_factory=list)   # per sample: [[x1,y1,x2,y2,score],...]
    cashier_zone: list[float] | None = None                        # [x1,y1,x2,y2] normalised

    def boxes_at(self, t: float) -> list[list[float]]:
        if not self.times:
            return []
        i = int(np.argmin([abs(x - t) for x in self.times]))
        return self.boxes[i]


def detect_people(store: FrameStore, cfg: dict, idx: list[int] | None = None) -> PersonTrack:
    """Person boxes on frames `idx` (default: the whole video at cfg fps)."""
    import torch

    if idx is None:
        idx = store.sample(0, store.duration, cfg.get("fps", 2.0))
    track = PersonTrack()
    with _lock:
        _load(cfg.get("model", "PekingU/rtdetr_r50vd"), cfg.get("device", "cpu"))
        for chunk_start in range(0, len(idx), 16):
            chunk = idx[chunk_start:chunk_start + 16]
            images = [store.image(i) for i in chunk]
            inputs = _processor(images=images, return_tensors="pt").to(_device)
            if _device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.no_grad():
                out = _model(**inputs)
            sizes = torch.tensor([[im.height, im.width] for im in images])
            results = _processor.post_process_object_detection(
                out, target_sizes=sizes, threshold=cfg.get("person_score", 0.5))
            for i, res in zip(chunk, results):
                persons = []
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    if int(label) == 0:   # COCO "person"
                        x1, y1, x2, y2 = box.float().tolist()
                        persons.append([x1 / store.width, y1 / store.height,
                                        x2 / store.width, y2 / store.height, float(score)])
                track.times.append(store.times[i])
                track.boxes.append(persons)
    if len(idx) > 10:
        track.cashier_zone = _estimate_cashier_zone(track)
    return track


def _estimate_cashier_zone(track: PersonTrack) -> list[float] | None:
    """Cell of a coarse grid occupied by a person most often, grown to a box."""
    gw, gh = 6, 10
    grid = np.zeros((gh, gw))
    for persons in track.boxes:
        seen = np.zeros_like(grid, dtype=bool)
        for x1, y1, x2, y2, _ in persons:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            seen[min(gh - 1, int(cy * gh)), min(gw - 1, int(cx * gw))] = True
        grid += seen
    if grid.max() == 0:
        return None
    r, c = np.unravel_index(int(grid.argmax()), grid.shape)
    # median box of people whose centre falls in the winning cell
    cands = []
    for persons in track.boxes:
        for x1, y1, x2, y2, _ in persons:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if int(cy * gh) == r and int(cx * gw) == c:
                cands.append([x1, y1, x2, y2])
    if not cands:
        return None
    return np.median(np.array(cands), axis=0).tolist()


def _iou_zone(box: list[float], zone: list[float]) -> float:
    x1, y1 = max(box[0], zone[0]), max(box[1], zone[1])
    x2, y2 = min(box[2], zone[2]), min(box[3], zone[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    return inter / area if area > 0 else 0.0


def rule_candidates(track: PersonTrack, cfg: dict) -> list[dict]:
    """Coarse candidates from person signals. They only point the VLM at moments worth checking."""
    cands: list[dict] = []
    if not track.times:
        return cands
    zone = track.cashier_zone
    counts = [len(p) for p in track.boxes]

    # 1. cashier zone becomes empty for a while -> possible "left workplace"
    if zone is not None:
        occupied = [any(_iou_zone(b, zone) > 0.3 for b in persons) for persons in track.boxes]
        min_abs = cfg.get("min_absence_sec", 3.0)
        i = 1
        while i < len(occupied):
            if occupied[i - 1] and not occupied[i]:
                j = i
                while j < len(occupied) and not occupied[j]:
                    j += 1
                end_t = track.times[j - 1]
                if end_t - track.times[i] >= min_abs or j == len(occupied):
                    cands.append({"type_id": "left_workplace_unsecured", "t": track.times[i - 1],
                                  "score": 0.3, "source": "rule:cashier_zone_empty"})
                i = j
            i += 1

    # 2. three or more people for a while (cashier + client + someone else) -> possible outsider
    min_crowd = cfg.get("crowd_min_people", 3)
    hold = max(1, int(round(cfg.get("crowd_min_sec", 2.0) * cfg.get("fps", 2.0))))
    last_t = -1e9
    for i in range(len(counts) - hold + 1):
        if all(c >= min_crowd for c in counts[i:i + hold]) and track.times[i] - last_t > 10:
            cands.append({"type_id": "outsider_in_cash_area", "t": track.times[i],
                          "score": 0.2, "source": "rule:crowd"})
            last_t = track.times[i]
    return cands
