"""Merging raw detections into candidate events and scoring them.

Evidence sources per candidate:
- `state`: workplace-state timeline rules (app/timeline.py) - the main detector;
- `scan`:  open-ended VLM scan windows (catches types the rules don't cover);
- `rule`:  person-detector heuristics (weak pointers);
- `verify`: a targeted yes/no re-check of the candidate window.

`score_candidate` and `select` are pure functions of the candidate features and the `fusion`
settings, so tools/tune.py can re-score cached runs without touching the GPU.
"""
from __future__ import annotations


def merge_raw(raw: list[dict], gap: float) -> list[dict]:
    """Group raw events by type and time proximity."""
    groups: list[dict] = []
    for ev in sorted(raw, key=lambda e: (e["type_id"], e["start"])):
        last = groups[-1] if groups else None
        if last and last["type_id"] == ev["type_id"] and ev["start"] <= last["end"] + gap:
            last["end"] = max(last["end"], ev["end"])
            last["items"].append(ev)
        else:
            groups.append({"type_id": ev["type_id"], "start": ev["start"], "end": ev["end"], "items": [ev]})

    cands = []
    for g in groups:
        scan = [e for e in g["items"] if e["source"] == "scan"]
        state = [e for e in g["items"] if e["source"].startswith("state")]
        rules = [e for e in g["items"] if e["source"].startswith("rule")]
        best_scan = max(scan, key=lambda e: e["confidence"]) if scan else None
        # onset estimate: state rule time first, then earliest confident scan hit, then rule time
        if state:
            t = min(e["start"] for e in state)
        elif scan:
            confident = [e for e in scan if e["confidence"] >= 0.5] or scan
            t = min(e["start"] for e in confident)
        else:
            t = min(e["start"] for e in rules)
        cands.append({
            "type_id": g["type_id"],
            "t": round(t, 2),
            "start": round(g["start"], 2),
            "end": round(g["end"], 2),
            "scan_conf": max((e["confidence"] for e in scan), default=0.0),
            "support": len({str(e.get("window")) for e in scan}),
            "state": max((e["confidence"] for e in state), default=0.0),
            "rule": max((e["confidence"] for e in rules), default=0.0),
            "sources": sorted({e["source"] for e in g["items"]}),
            "issues": sorted({x for e in state for x in e.get("issues", [])}),
            "evidence_t": next((e["evidence_t"] for e in state if e.get("evidence_t") is not None), None),
            "people": max((e["people"] for e in state if e.get("people") is not None), default=None),
            "what_is_visible": best_scan["what_is_visible"] if best_scan else "",
            "verify": None,     # {visible, confidence, first_frame_t, explanation_ru} from the verify pass
            "onset_t": None,    # refined onset (e.g. the moment the cashier stands up)
        })
    return cands


def base_score(c: dict, fcfg: dict) -> float:
    w = fcfg["weights"]
    s = max(w["scan"] * c["scan_conf"], c["state"], w["rule"] * c["rule"])
    s += w["support"] * max(0, min(c["support"], 3) - 1) / 2
    return min(1.0, s)


def score_candidate(c: dict, fcfg: dict) -> float:
    s = base_score(c, fcfg)
    v = c.get("verify")
    if v is not None:
        if v["visible"]:
            s = s + (1.0 - s) * fcfg["weights"]["verify"] * float(v["confidence"])
            s = max(s, fcfg.get("verify_alone", 0.8) * float(v["confidence"]))
        else:
            s *= fcfg.get("verify_fail_penalty", 0.6)
    if c["type_id"] == "outsider_in_cash_area" and c.get("people") is not None and c["people"] < 2:
        # only one person ever visible: it is the cashier herself (e.g. at the safe), not an outsider
        s *= fcfg.get("single_person_factor", 0.3)
    return round(min(1.0, s), 4)


def final_time(c: dict, fcfg: dict | None = None) -> float:
    """Reported timecode. State-rule candidates use their own time (first sample showing the
    violation) unless `use_onset` is set; otherwise the refined onset / verify first frame. Plus a
    per-type offset (`time_offset`, tuned on the labels)."""
    fcfg = fcfg or {}
    v = c.get("verify") or {}
    t = float(c["t"])
    if not c.get("state") or fcfg.get("use_onset", False):
        if c.get("onset_t") is not None:
            t = float(c["onset_t"])
        elif v.get("visible") and v.get("first_frame_t") is not None:
            t = float(v["first_frame_t"])
    return t + float((fcfg.get("time_offset") or {}).get(c["type_id"], 0.0))


def threshold_for(type_id: str, fcfg: dict) -> float:
    th = fcfg["thresholds"]
    return float(th.get(type_id, th["default"]))


def select(cands: list[dict], fcfg: dict) -> list[dict]:
    """Final violations: candidates over their type threshold, best first; `other` must be verified."""
    kept = []
    for c in cands:
        c["score"] = score_candidate(c, fcfg)
        if c["type_id"] == "other" and not (c.get("verify") or {}).get("visible"):
            continue
        if c["score"] >= threshold_for(c["type_id"], fcfg):
            kept.append(c)
    kept.sort(key=lambda c: -c["score"])
    # drop a weaker event of a different type at nearly the same moment (one action, two labels)
    final: list[dict] = []
    for c in kept:
        if any(abs(final_time(c, fcfg) - final_time(f, fcfg)) < 2.0 and c["score"] < f["score"] * 0.8 for f in final):
            continue
        final.append(c)
    return final[: int(fcfg.get("max_violations", 3))]
