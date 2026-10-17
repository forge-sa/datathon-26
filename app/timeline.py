"""Workplace-state timeline: sample the desk every few seconds, ask the VLM concrete questions
(who is at the desk, what lies on it, screen, safe), then derive violations from state changes.

This is the main detector: in this case the difference between a violation and normal work is
the *state* of the workplace (e.g. leaving with a lit work screen or money on the desk vs. leaving
after locking up), which a VLM reads far more reliably than an abstract "is this a violation?".
"""
from __future__ import annotations

from . import prompts
from .video import FrameStore, resize_for_vlm, to_jpeg

CASH_ITEMS = {"banknotes", "cash_bundle", "coins", "cash_bag", "package_or_envelope"}
PERSONAL_ITEMS = {"mobile_phone", "tablet", "cup_or_drink", "food", "handbag", "cosmetics"}


def sample_times(duration: float, cfg: dict) -> list[float]:
    step = max(cfg["every_sec"], duration / max(1, cfg["max_samples"]))
    times, t = [], min(1.0, duration / 2)
    while t < duration:
        times.append(round(t, 2))
        t += step
    return times


def desk_box(vlm, model: str, store: FrameStore, t: float) -> list[float]:
    i = store.index_at(t)
    out = vlm.chat_json(model, prompts.SYSTEM, [("room", to_jpeg(store.image(i)))],
                        prompts.GROUND_QUESTION, prompts.GROUND_SCHEMA)
    b = [float(v) for v in (out.get("bbox_2d") or [])[:4]]
    if len(b) == 4 and b[2] > b[0] and b[3] > b[1]:
        return [max(0.0, b[0] / 1000 - 0.06), max(0.0, b[1] / 1000 - 0.06),
                min(1.0, b[2] / 1000 + 0.06), min(1.0, b[3] / 1000 + 0.06)]
    return [0.0, 0.2, 1.0, 0.8]


def crop(store: FrameStore, i: int, box: list[float], width: int) -> bytes:
    img = store.image(i)
    w, h = img.size
    c = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    c = c.resize((width, max(32, int(c.height * width / max(1, c.width)))))
    return to_jpeg(c, 90)


def read_state(vlm, model: str, store: FrameStore, t: float, box: list[float], cfg: dict) -> dict:
    i = store.index_at(t)
    frames = [(f"whole room, t={store.times[i]:.1f}s", to_jpeg(resize_for_vlm(store.image(i), cfg["room_width"]))),
              (f"desk crop, t={store.times[i]:.1f}s", crop(store, i, box, cfg["crop_width"]))]
    out = vlm.chat_json(model, prompts.SYSTEM, frames, prompts.STATE_QUESTION, prompts.STATE_SCHEMA)
    objects = set(out.get("objects_on_desk") or [])
    return {
        "t": store.times[i],
        "cashier_at_desk": bool(out.get("cashier_at_desk")),
        "other_person_behind_desk": bool(out.get("other_person_behind_desk")),
        "objects": sorted(objects),
        "screen": out.get("screen", "not_visible"),
        "safe_open": bool(out.get("safe_or_drawer_open")),
    }


def unsecured(s: dict) -> list[str]:
    """What is left unsecured in one state sample (Russian phrases for the description)."""
    issues = []
    if s["screen"] == "application_window":
        issues.append("экран компьютера не заблокирован (открыто рабочее приложение)")
    cash = s["objects"] and CASH_ITEMS.intersection(s["objects"])
    if cash:
        issues.append("на столе остались деньги/ценности (" + ", ".join(sorted(cash)) + ")")
    dev = PERSONAL_ITEMS.intersection(s["objects"]) & {"mobile_phone", "tablet"}
    if dev:
        issues.append("на столе оставлен " + ("телефон" if "mobile_phone" in dev else "планшет"))
    if s["safe_open"]:
        issues.append("сейф или денежный ящик открыт")
    return issues


def derive(states: list[dict], cfg: dict) -> list[dict]:
    """Candidate violations from the state sequence. Each gets t, score and the evidence states."""
    cands: list[dict] = []
    n = len(states)

    # 1. cashier leaves the desk and something stays unsecured
    for k in range(1, n):
        if states[k - 1]["cashier_at_desk"] and not states[k]["cashier_at_desk"]:
            absent = []
            for s in states[k:k + 3]:
                if s["cashier_at_desk"]:
                    break
                absent.append(s)
            hits = [unsecured(s) for s in absent]
            n_hit = sum(1 for h in hits if h)
            if n_hit and (n_hit * 2 >= len(absent)):
                issues = sorted({x for h in hits for x in h})
                # t = first sample with the desk empty (labels mark the cashier being gone, not the
                # first movement); the onset check may refine it, see fusion.final_time
                cands.append({"type_id": "left_workplace_unsecured", "t": states[k]["t"],
                              "t_window": [states[k - 1]["t"], states[k]["t"]],
                              "score": min(0.95, 0.55 + 0.15 * n_hit), "issues": issues,
                              "source": "state:left_unsecured", "evidence_t": absent[0]["t"]})

    # 2. personal items (phone, cup, food, bag) on the desk while the cashier works
    def personal(s):
        return PERSONAL_ITEMS.intersection(s["objects"]) if s["cashier_at_desk"] else set()

    for run in _runs(states, personal):
        if len(run) >= cfg["personal_min_samples"]:
            found = sorted({x for r in run for x in personal(r)})
            cands.append({"type_id": "personal_items_phone", "t": run[0]["t"], "t_window": [run[0]["t"], run[-1]["t"]],
                          "score": min(0.9, 0.45 + 0.1 * len(run)), "issues": found,
                          "source": "state:personal_items", "evidence_t": run[0]["t"]})

    # 3. safe / cash drawer open for a long time
    for run in _runs(states, lambda s: s["safe_open"], gap=0):
        if len(run) >= cfg["safe_min_samples"]:
            cands.append({"type_id": "safe_left_open", "t": run[0]["t"], "t_window": [run[0]["t"], run[-1]["t"]],
                          "score": min(0.9, 0.4 + 0.1 * len(run)), "issues": ["сейф или денежный ящик открыт"],
                          "source": "state:safe_open", "evidence_t": run[0]["t"]})

    # 4. someone other than the cashier behind the desk (one sample alone is weak: often the
    # cashier herself standing at the safe, or a client leaning over the counter)
    for run in _runs(states, lambda s: s["other_person_behind_desk"]):
        cands.append({"type_id": "outsider_in_cash_area", "t": run[0]["t"], "t_window": [run[0]["t"], run[-1]["t"]],
                      "score": min(0.85, 0.15 + 0.2 * len(run)), "issues": [],
                      "source": "state:outsider", "evidence_t": run[0]["t"]})
    return cands


def _runs(states: list[dict], hit, gap: int = 1) -> list[list[dict]]:
    """Runs of samples where hit(s) is truthy, tolerating up to `gap` missed samples inside a run
    (a phone briefly hidden by a hand). Only the hit samples are returned."""
    runs, cur, miss = [], [], 0
    for s in states:
        if hit(s):
            cur.append(s)
            miss = 0
        elif cur:
            miss += 1
            if miss > gap:
                runs.append(cur)
                cur, miss = [], 0
    if cur:
        runs.append(cur)
    return runs
