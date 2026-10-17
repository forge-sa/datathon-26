"""Run the full pipeline over the labelled dataset and score it.

    python tools/eval.py --run baseline              # all 40 unique clips
    python tools/eval.py --run quick --clips X-11 X-23
    python tools/eval.py --run timing --deadline     # enforce the real 240 s job deadline

Per-clip outputs (incl. every candidate with its features) go to runs/<run>/<clip>.json, so
tools/tune.py can re-tune thresholds without running the VLM again. VLM calls are cached in
cache/vlm, so a re-run with unchanged prompts/frames is fast.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT, load_settings  # noqa: E402
from app.pipeline import analyze  # noqa: E402
from tools.labels import load_dataset  # noqa: E402
from tools.metrics import fmt, score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--clips", nargs="*", help="subset, e.g. X-11 X-23")
    ap.add_argument("--workers", type=int, default=1, help="clips processed in parallel")
    ap.add_argument("--deadline", action="store_true", help="enforce the per-job time budget")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    items = load_dataset()
    if args.clips:
        wanted = {c.replace("Х", "X").upper() for c in args.clips}
        items = [it for it in items if it["name"] in wanted]
    out_dir = ROOT / "runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    s = load_settings()
    if not args.deadline:
        s["budget"]["job_deadline_sec"] = 10_000

    def run(it):
        t0 = time.time()
        try:
            res = analyze(it["path"], settings=s, use_cache=not args.no_cache, active_jobs=args.workers)
        except Exception as e:  # noqa: BLE001
            print(f"{it['name']}: FAILED {e}")
            return it, None
        res["elapsed"] = round(time.time() - t0, 1)
        for v in res["violations"]:
            v["evidence_frames"] = [{"timestamp_sec": f["timestamp_sec"]} for f in v["evidence_frames"]]
        record = {"item": {k: (str(v) if isinstance(v, Path) else v) for k, v in it.items()}, **res}
        (out_dir / f"{it['name']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str),
                                                     encoding="utf-8")
        got = ", ".join(f"{v['type_id']}@{v['timecode_sec']}" for v in res["violations"]) or "none"
        want = f"violation@{it['t']}" + (f" ({it.get('type')})" if it.get("type") else "") if it["violation"] else "none"
        mark = "OK " if bool(res["violations"]) == it["violation"] else "XX "
        print(f"{mark}{it['name']:6} {res['elapsed']:6.1f}s  want: {want:45}  got: {got}", flush=True)
        return it, res

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, items))

    preds = {it["name"]: [{"t": v["timecode_sec"], "type_id": v["type_id"]} for v in res["violations"]]
             for it, res in results if res is not None}
    m = score([it for it, res in results if res is not None], preds)
    times = [res["elapsed"] for _, res in results if res]
    print("\n" + fmt(m))
    if times:
        print(f"  time per clip: mean {sum(times) / len(times):.1f}s  max {max(times):.1f}s")
    (out_dir / "_summary.json").write_text(json.dumps(m, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
