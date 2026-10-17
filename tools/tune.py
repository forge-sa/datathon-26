"""Post-tuning: re-score the cached candidates of an eval run under many fusion settings and
pick the best. No GPU needed - it only replays app.fusion on runs/<run>/*.json.

    python tools/tune.py --run baseline            # search, print best + leave-one-setup-out CV
    python tools/tune.py --run baseline --apply    # also write config/tuned.yaml (server picks it up)

Only fusion parameters are searched (thresholds, weights, verify penalty, timecode convention). Prompt / window
changes need a new eval run (tools/eval.py) because they change what the VLM sees.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import fusion  # noqa: E402
from app.config import CONFIG_DIR, ROOT, load_settings  # noqa: E402
from tools.labels import load_annotations  # noqa: E402
from tools.metrics import fmt, objective, score  # noqa: E402


def load_run(run: str) -> list[dict]:
    """Records of an eval run, with type/setup refreshed from the current data/annotations.yaml."""
    ann = load_annotations()
    recs = []
    for p in sorted((ROOT / "runs" / run).glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        r["item"].update(ann.get(r["item"]["name"], {}))
        recs.append(r)
    return recs


def predict(recs: list[dict], fcfg: dict) -> dict[str, list[dict]]:
    preds = {}
    for r in recs:
        cands = copy.deepcopy(r["debug"].get("candidates", []))
        final = fusion.select(cands, fcfg)
        preds[r["item"]["name"]] = [{"t": fusion.final_time(c, fcfg), "type_id": c["type_id"]} for c in final]
    return preds


TUNED_KEYS = ["thresholds", "weights", "verify_fail_penalty", "verify_alone", "use_onset", "time_offset"]
TYPE_THRESHOLDS = ["left_workplace_unsecured", "outsider_in_cash_area", "personal_items_phone", "safe_left_open"]


def grid(base: dict):
    for th, th_other, w_scan, w_verify, penalty, alone in itertools.product(
            [0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.8, 0.9],
            [0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7],
            [0.3, 0.5, 1.0],
            [0.6, 0.8]):
        f = copy.deepcopy(base)
        f["thresholds"] = {"default": th, "other": th_other}
        f["weights"] = {**f["weights"], "scan": w_scan, "verify": w_verify}
        f["verify_fail_penalty"] = penalty
        f["verify_alone"] = alone
        yield f


def search(recs: list[dict], base: dict) -> tuple[dict, dict]:
    """Global grid search, then per-type threshold refinement (coordinate descent)."""
    items = [r["item"] for r in recs]
    best = None
    for f in grid(base):
        m = score(items, predict(recs, f))
        if best is None or objective(m) > best[0]:
            best = (objective(m), f, m)
    _, f, m = best
    for type_id in TYPE_THRESHOLDS:
        for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]:
            g = copy.deepcopy(f)
            g["thresholds"][type_id] = th
            mg = score(items, predict(recs, g))
            if objective(mg) > objective(m) + 1e-9:
                f, m = g, mg
    # timecode convention: state time vs refined onset, then a per-type offset
    for use_onset in [False, True]:
        g = copy.deepcopy(f)
        g["use_onset"] = use_onset
        mg = score(items, predict(recs, g))
        if objective(mg) > objective(m) + 1e-9:
            f, m = g, mg
    for type_id in TYPE_THRESHOLDS:
        for off in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
            g = copy.deepcopy(f)
            g["time_offset"] = {**(g.get("time_offset") or {}), type_id: off}
            mg = score(items, predict(recs, g))
            if objective(mg) > objective(m) + 1e-9:
                f, m = g, mg
    return f, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="latest")
    ap.add_argument("--apply", action="store_true", help="write the best fusion settings to config/tuned.yaml")
    args = ap.parse_args()

    recs = load_run(args.run)
    if not recs:
        sys.exit(f"no records in runs/{args.run}; run tools/eval.py --run {args.run} first")
    base = load_settings()["fusion"]
    items = [r["item"] for r in recs]

    print(f"current settings on {len(recs)} clips:\n  " + fmt(score(items, predict(recs, base))))
    best, m = search(recs, base)
    print("\nbest on all clips (optimistic, fitted on the same data):\n  " + fmt(m))
    print("  params:", json.dumps({k: best.get(k) for k in TUNED_KEYS}))

    setups = sorted({r["item"].get("setup") for r in recs if r["item"].get("setup")})
    if len(setups) >= 2:
        # leave-one-setup-out: tune on the other camera setups, test on the held-out one
        preds_cv = {}
        for s in setups:
            train = [r for r in recs if r["item"].get("setup") != s]
            test = [r for r in recs if r["item"].get("setup") == s]
            f, _ = search(train, base)
            preds_cv.update(predict(test, f))
        held = [r["item"] for r in recs if r["item"].get("setup")]
        print(f"\nleave-one-setup-out CV over {len(setups)} setups (honest estimate for a new angle):\n  "
              + fmt(score(held, preds_cv)))
    else:
        print("\n(no `setup` in data/annotations.yaml - run tools/setups.py --write for leave-one-setup-out CV)")

    if args.apply:
        out = {"fusion": {k: best.get(k) for k in TUNED_KEYS}}
        path = CONFIG_DIR / "tuned.yaml"
        path.write_text("# written by tools/tune.py --run " + args.run + "; delete to fall back to settings.yaml\n"
                        + yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
