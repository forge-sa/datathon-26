"""Export scan windows from the labelled clips as a supervised fine-tuning set for Qwen3-VL.

    python tools/export_sft.py --from-run baseline

Writes data/sft/train.jsonl and data/sft/val.jsonl (val = one camera setup, or every 5th clip),
plus JPEG frames under data/sft/images/. Each record is one scan window in exactly the format
the service sends at inference time (same prompt, same frame size), so a LoRA trained on it
plugs straight into the pipeline.

Targets:
- negative clips: every window -> no events;
- positive clips: windows that contain the labelled onset -> one event of the annotated type;
  windows fully before the onset -> no events; windows after the onset are skipped (the label
  only marks the onset, so they are ambiguous).
`scene` text is taken from an earlier eval run (--from-run) when available, otherwise empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import prompts  # noqa: E402
from app.config import ROOT, load_settings, load_taxonomy  # noqa: E402
from app.pipeline import plan_windows  # noqa: E402
from app.video import decode, resize_for_vlm  # noqa: E402
from tools.labels import load_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", default=None, help="reuse `scene` texts from runs/<run>")
    ap.add_argument("--out", default=str(ROOT / "data" / "sft"))
    ap.add_argument("--val-setup", default=None, help="camera setup held out for validation")
    args = ap.parse_args()

    s = load_settings()
    types = load_taxonomy()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    question_cache: dict[int, str] = {}
    files = {"train": open(out / "train.jsonl", "w", encoding="utf-8"),
             "val": open(out / "val.jsonl", "w", encoding="utf-8")}
    counts = {"train": [0, 0], "val": [0, 0]}

    for n, it in enumerate(load_dataset()):
        split = "val" if ((args.val_setup and it.get("setup") == args.val_setup)
                          or (not args.val_setup and n % 5 == 4)) else "train"
        scenes = {}
        if args.from_run:
            p = ROOT / "runs" / args.from_run / f"{it['name']}.json"
            if p.exists():
                rec = json.loads(p.read_text(encoding="utf-8"))
                scenes = {tuple(w["window"]): w.get("scene", "") for w in rec["debug"].get("windows", [])}

        store = decode(it["path"], fps=s["decode"]["fps"])
        onsets = [t for t in [it["t"]] + it.get("extra_t", []) if t is not None] if it["violation"] else []
        vtype = it.get("type") if it.get("type") in types else "other"
        for a, b in sorted(plan_windows(store.duration, s["scan"]["window_sec"], s["scan"]["stride_sec"])):
            inside = [t for t in onsets if a <= t <= b]
            after = any(t < a for t in onsets)
            if after and not inside:
                continue
            idx = store.sample(a, b, s["scan"]["fps"])
            paths, labels = [], []
            for k, i in enumerate(idx):
                p = out / "images" / f"{it['name']}_{store.times[i]:07.2f}.jpg"
                if not p.exists():
                    resize_for_vlm(store.image(i), s["scan"]["frame_width"]).save(p, quality=85)
                paths.append(str(p.relative_to(out)))
                labels.append(f"frame {k + 1}, t={store.times[i]:.1f}s")
            events = []
            for t in inside:
                k1 = min(range(len(idx)), key=lambda k: abs(store.times[idx[k]] - t)) + 1
                events.append({"type_id": vtype, "start_frame": k1, "end_frame": len(idx), "confidence": 0.9,
                               "what_is_visible": types[vtype]["definition"].split(":")[0].strip()})
            target = {"scene": scenes.get((round(a, 2), round(b, 2)), ""), "events": events}
            q = question_cache.setdefault(len(idx), prompts.scan_question(types, len(idx)))
            rec = {"clip": it["name"], "window": [a, b], "system": prompts.SYSTEM, "images": paths,
                   "frame_labels": labels, "question": q, "target": json.dumps(target, ensure_ascii=False)}
            files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split][bool(events)] += 1
        print(f"{it['name']:6} -> {split}")

    for f in files.values():
        f.close()
    for k, (neg, pos) in counts.items():
        print(f"{k}: {pos} positive windows, {neg} negative windows")


if __name__ == "__main__":
    main()
