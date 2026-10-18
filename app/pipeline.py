"""End-to-end analysis of one video. Everything respects a per-job deadline.

decode -> person boxes -> desk location -> workplace-state timeline (main detector)
       -> open-ended VLM scan (unseen types) -> fusion -> targeted checks (onset / verify)
       -> evidence frames + Russian descriptions with verbatim regulation quotes
"""
from __future__ import annotations

import base64
import logging
import math
import time
from pathlib import Path

from PIL import ImageDraw

from . import fusion, prompts, timeline
from .config import load_regulations, load_settings, load_taxonomy
from .video import FrameStore, decode, resize_for_vlm, to_jpeg
from .vlm import LATENCY, WALL, VLMClient, VLMError

log = logging.getLogger("pipeline")

ITEM_RU = {
    "mobile_phone": "мобильный телефон", "tablet": "планшет", "cup_or_drink": "чашка/напиток", "food": "еда",
    "handbag": "сумка", "cosmetics": "косметика", "banknotes": "банкноты", "cash_bundle": "пачка денег",
    "coins": "монеты", "cash_bag": "денежная сумка", "package_or_envelope": "пакет/конверт",
}


class Budget:
    def __init__(self, started_at: float, deadline_sec: float, reserve_sec: float):
        self.started_at = started_at
        self.deadline = started_at + deadline_sec
        self.reserve = reserve_sec

    def left(self) -> float:
        return self.deadline - self.reserve - time.time()

    def can_call(self, s: dict, n: int = 1) -> bool:
        return self.left() > WALL.get(s["vlm"]["sec_per_call_init"]) * n


def plan_windows(duration: float, window: float, stride: float) -> list[tuple[float, float]]:
    if duration <= window:
        return [(0.0, round(duration, 2))]
    n = int(math.ceil((duration - window) / stride)) + 1
    wins = [(min(k * stride, duration - window), min(k * stride, duration - window) + window) for k in range(n)]
    return sorted(set((round(a, 2), round(b, 2)) for a, b in wins))


def estimate_calls(s: dict, tier: dict, duration: float) -> int:
    tl = {**s["timeline"], "every_sec": tier["state_every_sec"]}
    calls = 1 + len(timeline.sample_times(duration, tl)) * (2 if tier.get("hands", True) else 1)
    if tier["scan"]:
        calls += len(plan_windows(duration, s["scan"]["window_sec"], s["scan"]["stride_sec"]))
    if tier["verify"]:
        calls += 3
    return calls


def choose_tier(s: dict, duration: float, budget: Budget, active_jobs: int) -> dict:
    """Cheapest-first fallback: the first tier whose estimated GPU time fits the budget.
    With one Ollama slot, concurrent jobs share the GPU, so their calls add up."""
    spc = LATENCY.get(s["vlm"]["sec_per_call_init"])
    load = max(1.0, active_jobs / max(1, s.get("ollama_parallel", 1)))
    for tier in s["tiers"]:
        if estimate_calls(s, tier, duration) * spc * load <= budget.left() * 0.85:
            return tier
    return s["tiers"][-1]


def analyze(video_path: str | Path, *, started_at: float | None = None, active_jobs: int = 1,
            settings: dict | None = None, progress=lambda msg: None, use_cache: bool = True,
            on_partial=None) -> dict:
    """`on_partial(result)` (optional) receives a provisional result after the main detector, so the
    job watchdog has something real to return if a later stage overruns the jury's 5-minute limit."""
    s = settings or load_settings()
    types = load_taxonomy()
    regs = load_regulations()
    budget = Budget(started_at or time.time(), s["budget"]["job_deadline_sec"], s["budget"]["reserve_sec"])
    vlm = VLMClient(s["vlm"], use_cache=use_cache)
    model = s["vlm"]["model"]
    debug: dict = {"timings": {}, "windows": [], "states": [], "warnings": []}
    t0 = time.time()

    progress("Декодирование видео")
    store = decode(video_path, fps=s["decode"]["fps"])
    debug["video"] = {"duration": round(store.duration, 2), "width": store.width, "height": store.height,
                      "frames_kept": len(store.times)}
    debug["timings"]["decode"] = round(time.time() - t0, 2)

    # -- person heuristics over the whole video (optional, off by default) ----------------
    # With rules off, person boxes are computed later only for the evidence frames.
    track = None
    raw: list[dict] = []
    if s["perception"].get("enabled") and s["perception"].get("rules"):
        progress("Детекция людей")
        t1 = time.time()
        try:
            from .perception import detect_people, rule_candidates
            track = detect_people(store, s["perception"])
            for c in rule_candidates(track, s["perception"]):
                raw.append({"type_id": c["type_id"], "start": c["t"], "end": c["t"], "confidence": c["score"],
                            "source": c["source"]})
            debug["cashier_zone"] = track.cashier_zone
        except Exception as e:  # perception is an aid; the VLM passes still run without it
            log.exception("perception failed")
            debug["warnings"].append(f"perception failed: {e}")
        debug["timings"]["perception"] = round(time.time() - t1, 2)

    tier = choose_tier(s, store.duration, budget, active_jobs)
    debug["tier"] = tier["name"]
    if tier.get("model"):
        model = tier["model"]

    # -- workplace-state timeline (main detector) --------------------------------------
    t2 = time.time()
    box = [0.0, 0.2, 1.0, 0.8]
    states: list[dict] = []
    try:
        progress("Поиск рабочего места кассира")
        box = timeline.desk_box(vlm, model, store, min(1.0, store.duration / 2))
        tl_cfg = {**s["timeline"], "every_sec": tier["state_every_sec"]}
        times = timeline.sample_times(store.duration, tl_cfg)
        for n, t in enumerate(times, 1):
            if not budget.can_call(s, 3):
                debug["warnings"].append(f"state timeline cut at {t:.1f}s (time budget)")
                break
            progress(f"Состояние рабочего места: {n}/{len(times)}")
            try:
                st = timeline.read_state(vlm, model, store, t, box, s["timeline"])
            except VLMError as e:   # one unreadable sample must not end the timeline
                debug["warnings"].append(f"state at {t:.1f}s skipped: {str(e)[:120]}")
                continue
            if tier.get("hands", True) and st["cashier_at_desk"]:
                try:
                    st["hands"] = timeline.read_hands(vlm, model, store, t, box, s["timeline"])
                except VLMError as e:
                    debug["warnings"].append(f"hands at {t:.1f}s skipped: {str(e)[:120]}")
            states.append(st)
    except VLMError as e:
        debug["warnings"].append(f"state timeline failed: {e}")
    debug["desk_box"] = box
    debug["states"] = states
    for c in timeline.derive(states, s["timeline"]):
        if c["type_id"] == "outsider_in_cash_area" and s["perception"].get("enabled"):
            _count_people(store, c, s["perception"], debug)
        raw.append({"type_id": c["type_id"], "start": c["t"], "end": c["t_window"][1], "confidence": c["score"],
                    "source": c["source"], "issues": c["issues"], "evidence_t": c["evidence_t"],
                    "people": c.get("people")})
    debug["timings"]["timeline"] = round(time.time() - t2, 2)
    if on_partial is not None:
        try:
            prov = fusion.select(fusion.merge_raw([dict(r) for r in raw], s["fusion"]["merge_gap_sec"]), s["fusion"])
            on_partial(_result(prov, store, types, regs, s, None, with_boxes=False))
        except Exception:  # the provisional result is a safety net only
            log.exception("provisional result failed")

    # -- open-ended scan (types the state rules don't cover) ----------------------------
    t3 = time.time()
    if tier["scan"]:
        windows = plan_windows(store.duration, s["scan"]["window_sec"], s["scan"]["stride_sec"])
        schema = prompts.scan_schema(list(types.keys()))
        for n, (a, b) in enumerate(windows, 1):
            if not budget.can_call(s, 3):
                debug["warnings"].append(f"{len(windows) - n + 1} scan windows skipped (time budget)")
                break
            progress(f"Общий просмотр: окно {n}/{len(windows)}")
            idx = store.sample(a, b, s["scan"]["fps"])
            try:
                out = vlm.chat_json(model, prompts.SYSTEM, _frames(store, idx, s["scan"]["frame_width"]),
                                    prompts.scan_question(types, len(idx)), schema)
            except VLMError as e:
                debug["warnings"].append(f"scan window failed: {e}")
                continue
            debug["windows"].append({"window": [a, b], "scene": out.get("scene", ""), "events": out.get("events", []),
                                     "latency": out.get("_latency"), "cached": out.get("_cached", False)})
            for ev in out.get("events", []):
                if ev.get("type_id") not in types:
                    continue
                k1 = min(max(int(ev.get("start_frame", 1)), 1), len(idx))
                k2 = min(max(int(ev.get("end_frame", k1)), k1), len(idx))
                raw.append({"type_id": ev["type_id"], "start": store.times[idx[k1 - 1]],
                            "end": store.times[idx[k2 - 1]],
                            "confidence": max(0.0, min(1.0, float(ev.get("confidence", 0.5)))),
                            "what_is_visible": ev.get("what_is_visible", ""), "source": "scan", "window": [a, b]})
    debug["timings"]["scan"] = round(time.time() - t3, 2)

    # -- fusion + targeted checks -------------------------------------------------------
    fcfg = s["fusion"]
    cands = fusion.merge_raw(raw, fcfg["merge_gap_sec"])
    cands.sort(key=lambda c: -fusion.base_score(c, fcfg))
    t4 = time.time()
    if tier["verify"]:
        todo = [c for c in cands if fusion.base_score(c, fcfg) >= fcfg["candidate_min_score"]]
        todo = todo[: s["verify"]["max_events"]]
        progress(f"Проверка кандидатов: {len(todo)}")
        for c in todo:
            if not budget.can_call(s, 2):
                debug["warnings"].append("targeted checks cut short (time budget)")
                break
            try:
                _check(vlm, model, s, store, types[c["type_id"]], c, box)
            except VLMError as e:
                debug["warnings"].append(f"check failed: {e}")
    debug["timings"]["verify"] = round(time.time() - t4, 2)

    final = fusion.select(cands, fcfg)
    debug["candidates"] = cands

    # -- output -------------------------------------------------------------------------
    out = _result(final, store, types, regs, s, track)
    debug["timings"]["total"] = round(time.time() - budget.started_at, 2)
    out["debug"] = debug
    return out


def _result(final: list[dict], store: FrameStore, types: dict, regs: dict, s: dict, track,
            with_boxes: bool = True) -> dict:
    fcfg = s["fusion"]
    violations = []
    for c in final:
        t = round(max(0.0, min(store.duration, fusion.final_time(c, fcfg))), 1)
        tdef = types[c["type_id"]]
        violations.append({
            "type": tdef["name_ru"],
            "timecode_sec": t,
            "description": describe(tdef, c, regs, fcfg),
            "evidence_frames": evidence_frames(store, t, track, c.get("evidence_t"),
                                               s["perception"] if with_boxes else None),
            "type_id": c["type_id"],
            "confidence": c["score"],
            "rules": [cite(regs, cid) for cid in tdef.get("clauses", []) if cid in regs["clauses"]],
        })
    violations.sort(key=lambda v: v["timecode_sec"])
    return {"verdict": "violation" if violations else "no_violation", "violations": violations}


def _count_people(store: FrameStore, c: dict, pcfg: dict, debug: dict) -> None:
    """An "outsider" needs a second person. The VLM often calls the cashier herself an outsider when she
    stands up and goes to the safe (measured: 4 of 6 false alarms in run tl6), so count people with the
    person detector over the run."""
    a, b = c["t_window"]
    times = [a + (b - a) * k / 4 for k in range(5)] if b > a else [a - 0.5, a, a + 0.5]
    idx = sorted({store.index_at(max(0.0, min(store.duration, t))) for t in times})
    try:
        from .perception import detect_people
        track = detect_people(store, pcfg, idx=idx)
    except Exception as e:  # noqa: BLE001 - perception is an aid only
        debug["warnings"].append(f"people count failed: {e}")
        return
    # reflections in the glass door / people behind it score ~0.55-0.6, people in the room >= 0.89
    min_h = float(pcfg.get("min_person_height", 0.12))
    min_score = float(pcfg.get("count_score", 0.8))
    counts = [sum(1 for x1, y1, x2, y2, sc in boxes if y2 - y1 >= min_h and sc >= min_score)
              for boxes in track.boxes]
    c["people"] = max(counts, default=0)    # fusion.score_candidate weakens single-person candidates


def _frames(store: FrameStore, idx: list[int], width: int) -> list[tuple[str, bytes]]:
    return [(f"frame {k + 1}, t={store.times[i]:.1f}s", to_jpeg(resize_for_vlm(store.image(i), width)))
            for k, i in enumerate(idx)]


def _check(vlm: VLMClient, model: str, s: dict, store: FrameStore, tdef: dict, c: dict, box: list[float]) -> None:
    """Targeted follow-up for one candidate; fills c['onset_t'] and/or c['verify']."""
    vc = s["verify"]
    if tdef.get("check") == "desk_state":
        # when exactly does the cashier stand up? (label convention for "left the workplace")
        a = c["start"] - vc["pad_before_sec"]
        b = (c["evidence_t"] if c.get("evidence_t") is not None else c["end"]) + 1.0
        idx = store.sample(a, b, vc["fps"])
        out = vlm.chat_json(model, prompts.SYSTEM, _frames(store, idx, vc["frame_width"]),
                            prompts.ONSET_QUESTION, prompts.ONSET_SCHEMA)
        k = int(out.get("first_frame", 0) or 0)
        if out.get("left") and 1 <= k <= len(idx):
            c["onset_t"] = store.times[idx[k - 1]]
        if not c["issues"]:
            # scan-only candidate: read the desk state after leaving to see what was left behind
            t_after = min(store.duration, (c["onset_t"] or c["t"]) + 4.0)
            st = timeline.read_state(vlm, model, store, t_after, box, s["timeline"])
            issues = timeline.unsecured(st) if not st["cashier_at_desk"] else []
            c["issues"] = issues
            c["evidence_t"] = st["t"]
            c["verify"] = {"visible": bool(issues), "confidence": 0.8 if issues else 0.0,
                           "first_frame_t": c["onset_t"], "explanation_ru": "", "state": st}
        return

    if tdef.get("check") == "none":
        return
    if tdef.get("check") == "personal":
        # close-ups of the cashier around the candidate: a phone or a cup is a few pixels in the full frame
        times = [c["t"] - 0.8, c["t"], c["t"] + 0.8, c["t"] + 1.6]
        idx = sorted({store.index_at(max(0.0, min(store.duration, x))) for x in times})
        frames = [(f"frame {k + 1}, t={store.times[i]:.1f}s", timeline.crop(store, i, timeline.wide_box(box), 640))
                  for k, i in enumerate(idx)]
        out = vlm.chat_json(model, prompts.SYSTEM, frames, prompts.PERSONAL_CHECK_QUESTION,
                            prompts.PERSONAL_CHECK_SCHEMA)
    else:
        idx = store.sample(c["t"] - vc["pad_before_sec"], c["t"] + vc["pad_after_sec"], vc["fps"])
        out = vlm.chat_json(model, prompts.SYSTEM, _frames(store, idx, vc["frame_width"]),
                            prompts.verify_question(tdef, len(idx)), prompts.VERIFY_SCHEMA)
    k = int(out.get("first_frame", 0) or 0)
    c["verify"] = {
        "visible": bool(out.get("visible")),
        "confidence": max(0.0, min(1.0, float(out.get("confidence", 0.0)))),
        "first_frame_t": store.times[idx[k - 1]] if 1 <= k <= len(idx) else None,
        "explanation_ru": (out.get("explanation_ru") or "").strip(),
        "observation": out.get("observation", ""),
        "window": [store.times[idx[0]], store.times[idx[-1]]],
    }


def cite(regs: dict, clause_id: str) -> dict:
    cl = regs["clauses"][clause_id]
    src = regs["sources"].get(cl["source"], {})
    return {"act": src.get("title", cl["source"]), "point": cl.get("point", ""), "quote": cl.get("quote", ""),
            "url": src.get("url", "")}


def _seen_text(c: dict, fcfg: dict) -> str:
    t = fusion.final_time(c, fcfg)
    issues = c.get("issues") or []
    for en, ru in ITEM_RU.items():   # object ids inside issue phrases -> Russian
        issues = [x.replace(en, ru) for x in issues]
    if c["type_id"] == "left_workplace_unsecured" and issues:
        return f"около {t:.1f} с кассир встаёт и уходит с рабочего места; после ухода {', '.join(issues)}"
    if c["type_id"] == "personal_items_phone" and issues:
        acts = [x for x in issues if " " in x]              # phrases from the hands rules
        items = [ITEM_RU.get(x, x) for x in issues if " " not in x]
        parts = acts + ([f"на столе кассира видны личные предметы ({', '.join(items)})"] if items else [])
        return f"{c['start']:.0f}–{c['end']:.0f} с: " + "; ".join(parts)
    if c["type_id"] == "cash_left_on_desk" and "state:loose_cash" in c.get("sources", []):
        return f"с {c['start']:.0f} по {c['end']:.0f} с на столе открыто лежат наличные, с которыми никто не работает"
    if c["type_id"] == "safe_left_open" and "state:safe_open" in c.get("sources", []):
        return f"сейф или денежный ящик стоит открытым с {c['start']:.0f} по {c['end']:.0f} с"
    v = (c.get("verify") or {}).get("explanation_ru", "").strip()
    if v:
        return v
    if c.get("what_is_visible"):
        return f"модель отметила: {c['what_is_visible'].strip()}"
    return ""


def describe(tdef: dict, c: dict, regs: dict, fcfg: dict) -> str:
    seen = _seen_text(c, fcfg)
    parts = [f"Возможное нарушение: {tdef['name_ru'].lower()}."]
    if seen:
        parts.append(f"Что видно: {seen.rstrip('.')}.")
    parts.append(f"Почему это может быть нарушением: {tdef['why_ru']}")
    cites = [cite(regs, cid) for cid in tdef.get("clauses", []) if cid in regs["clauses"]]
    if cites:
        refs = "; ".join(f"{x['act']}, {x['point']}" + (f": «{x['quote']}»" if x["quote"] else "") for x in cites)
        parts.append(f"Основание: {refs}.")
    parts.append("Результат является подсказкой для проверки человеком.")
    return " ".join(parts)


def evidence_frames(store: FrameStore, t: float, track, evidence_t: float | None = None,
                    pcfg: dict | None = None) -> list[dict]:
    """Up to 3 frames: the event moment, the frame that shows the state (if different), and context."""
    wanted = [t]
    if evidence_t is not None and abs(evidence_t - t) >= 1.0:
        wanted.append(evidence_t)
    wanted += [t + 2.5, t - 2.0]
    picked: list[int] = []
    for tt in wanted:
        i = store.index_at(max(0.0, min(store.duration, tt)))
        if i not in picked:
            picked.append(i)
        if len(picked) == 3:
            break
    if track is None and pcfg and pcfg.get("enabled"):
        try:
            from .perception import detect_people
            track = detect_people(store, pcfg, idx=picked)
        except Exception:
            log.exception("person boxes for evidence frames failed")
    out = []
    for i in picked:
        img = store.image(i)
        draw = ImageDraw.Draw(img)
        if track is not None:
            for x1, y1, x2, y2, _ in track.boxes_at(store.times[i]):
                draw.rectangle([x1 * img.width, y1 * img.height, x2 * img.width, y2 * img.height],
                               outline=(0, 255, 120), width=2)
        draw.rectangle([0, 0, 110, 22], fill=(0, 0, 0))
        draw.text((6, 5), f"t = {store.times[i]:.1f} s", fill=(255, 255, 255))
        out.append({"timestamp_sec": round(store.times[i], 1),
                    "image_base64": base64.b64encode(to_jpeg(img, 88)).decode()})
    return out
