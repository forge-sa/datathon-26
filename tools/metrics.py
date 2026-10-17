"""Scoring predictions against labels (shared by eval.py and tune.py)."""
from __future__ import annotations


def score(items: list[dict], preds: dict[str, list[dict]]) -> dict:
    """items: dataset entries; preds: name -> list of {"t", "type_id"} final violations."""
    tp = fn = fp = tn = 0
    dt, within1, within2, type_ok, type_n = [], 0, 0, 0, 0
    misses, false_alarms = [], []
    for it in items:
        p = preds.get(it["name"], [])
        if it["violation"]:
            if p:
                tp += 1
                label_ts = [it["t"]] + it.get("extra_t", [])
                best = min(((abs(v["t"] - lt), v) for v in p for lt in label_ts if lt is not None),
                           key=lambda x: x[0], default=(None, None))
                if best[0] is not None:
                    dt.append(best[0])
                    within1 += best[0] <= 1.0
                    within2 += best[0] <= 2.0
                    if it.get("type") and it["type"] != "unknown":
                        type_n += 1
                        type_ok += best[1]["type_id"] == it["type"]
            else:
                fn += 1
                misses.append(it["name"])
        else:
            if p:
                fp += 1
                false_alarms.append(it["name"])
            else:
                tn += 1
    pos, neg = tp + fn, fp + tn
    return {
        "n": len(items), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / pos if pos else 0.0,
        "specificity": tn / neg if neg else 0.0,
        "accuracy": (tp + tn) / len(items) if items else 0.0,
        "mean_abs_dt": sum(dt) / len(dt) if dt else None,
        "within_1s": within1 / tp if tp else 0.0,
        "within_2s": within2 / tp if tp else 0.0,
        "type_acc": type_ok / type_n if type_n else None,
        "misses": misses, "false_alarms": false_alarms,
    }


def objective(m: dict) -> float:
    """Misses are unacceptable, false alarms should be few: recall dominates."""
    return 1.5 * m["recall"] + 1.0 * m["specificity"] + 0.3 * m["within_2s"] + 0.2 * (m["type_acc"] or 0.0)


def fmt(m: dict) -> str:
    dt = f"{m['mean_abs_dt']:.2f}s" if m["mean_abs_dt"] is not None else "-"
    ta = f"{m['type_acc']:.0%}" if m["type_acc"] is not None else "-"
    return (f"recall {m['recall']:.0%} ({m['tp']}/{m['tp'] + m['fn']})  "
            f"specificity {m['specificity']:.0%} ({m['tn']}/{m['tn'] + m['fp']})  acc {m['accuracy']:.0%}  "
            f"|dt| {dt}  ≤1s {m['within_1s']:.0%}  ≤2s {m['within_2s']:.0%}  type {ta}\n"
            f"  misses: {', '.join(m['misses']) or '-'}\n  false alarms: {', '.join(m['false_alarms']) or '-'}")
