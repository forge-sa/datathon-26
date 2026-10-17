"""Group clips into camera setups by comparing their median frames.

    python tools/setups.py           # print clusters
    python tools/setups.py --write   # store `setup: S1..` per clip in data/annotations.yaml

Check the result by eye (it writes a contact sheet to runs/setups.jpg).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT  # noqa: E402
from app.video import decode  # noqa: E402
from tools.labels import ANNOTATIONS, load_dataset  # noqa: E402


def signature(path: Path) -> tuple[np.ndarray, Image.Image]:
    store = decode(path, fps=0.5)
    frames = [np.asarray(store.image(i).convert("L").resize((24, 44)), dtype=np.float32) for i in range(len(store.times))]
    med = np.median(np.stack(frames), axis=0)
    v = (med - med.mean()) / (med.std() + 1e-6)
    return v.ravel(), store.image(len(store.times) // 2).resize((116, 212))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.75, help="correlation to join a setup")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    items = load_dataset()
    sigs, thumbs = {}, {}
    for it in items:
        sigs[it["name"]], thumbs[it["name"]] = signature(it["path"])

    clusters: list[list[str]] = []
    for name, v in sigs.items():
        best, best_c = None, -1.0
        for k, members in enumerate(clusters):
            c = float(np.mean([np.dot(v, sigs[m]) / len(v) for m in members]))
            if c > best_c:
                best, best_c = k, c
        if best is not None and best_c >= args.threshold:
            clusters[best].append(name)
        else:
            clusters.append([name])

    labels = {}
    for k, members in enumerate(clusters, 1):
        print(f"S{k}: {', '.join(members)}")
        for m in members:
            labels[m] = f"S{k}"

    # contact sheet: one row per setup
    rows = [Image.new("RGB", (116 * max(len(c) for c in clusters), 212), "white") for _ in clusters]
    for r, members in zip(rows, clusters):
        for j, m in enumerate(members):
            r.paste(thumbs[m], (116 * j, 0))
    sheet = Image.new("RGB", (rows[0].width, 212 * len(rows)), "white")
    for i, r in enumerate(rows):
        sheet.paste(r, (0, 212 * i))
    (ROOT / "runs").mkdir(exist_ok=True)
    sheet.save(ROOT / "runs" / "setups.jpg", quality=80)
    print("contact sheet: runs/setups.jpg")

    if args.write:
        data = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8")) if ANNOTATIONS.exists() else {}
        clips = data.setdefault("clips", {}) or {}
        for name, s in labels.items():
            clips.setdefault(name, {})
            clips[name] = {**(clips[name] or {}), "setup": s}
        data["clips"] = dict(sorted(clips.items(), key=lambda kv: int(kv[0].split("-")[1])))
        ANNOTATIONS.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
        print(f"wrote setups into {ANNOTATIONS} (comments in that file are not preserved)")


if __name__ == "__main__":
    main()
