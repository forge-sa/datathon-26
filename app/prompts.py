"""Prompts and JSON schemas for the VLM passes."""
from __future__ import annotations

SYSTEM = (
    "You are a careful video auditor for a bank. You watch frames from a single fixed overhead "
    "security camera above a bank cashier's workplace (cash desk, safe, computer, client side of "
    "the counter). People in the scene: the cashier, clients, sometimes cash collectors or other "
    "staff. You report only what is clearly visible in the frames. Never identify people, never "
    "guess hidden information (system records, amounts). If unsure, report nothing."
)


def taxonomy_block(types: dict[str, dict]) -> str:
    lines = []
    for t in types.values():
        lines.append(f"- {t['id']}: {t['definition'].strip()}\n  NOT a violation: {t['not_violation'].strip()}")
    return "\n".join(lines)


def scan_question(types: dict[str, dict], n_frames: int) -> str:
    return (
        "Watch the frames and decide whether any of these cash-discipline violations happens:\n"
        f"{taxonomy_block(types)}\n\n"
        "First write `scene`: a short factual description of what happens across the frames "
        "(who is where, what they do with cash, safe, computer, bags, phones). Then list `events`. "
        f"Frame numbers go from 1 to {n_frames}. For each event give the type_id, the first frame "
        "where the violation becomes visible (for leaving the workplace: the frame where the cashier "
        "stands up / starts to leave), the last frame, a confidence between 0 and 1, and what is "
        "visible. Return an empty `events` list when everything is normal - this is the most "
        "common case."
    )


def scan_schema(type_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "scene": {"type": "string"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type_id": {"type": "string", "enum": type_ids},
                        "start_frame": {"type": "integer"},
                        "end_frame": {"type": "integer"},
                        "confidence": {"type": "number"},
                        "what_is_visible": {"type": "string"},
                    },
                    "required": ["type_id", "start_frame", "end_frame", "confidence", "what_is_visible"],
                },
            },
        },
        "required": ["scene", "events"],
    }


def verify_question(t: dict, n_frames: int) -> str:
    return (
        "Check one specific suspected violation.\n"
        f"Suspected violation: {t['definition'].strip()}\n"
        f"This is NOT a violation: {t['not_violation'].strip()}\n\n"
        "Look carefully at every frame. Answer:\n"
        "- `observation`: first describe in English who is where and what they do (cashier side vs "
        "client side of the counter, safe, cash, bags, phones);\n"
        "- `visible`: true only if the violation is clearly visible in these frames;\n"
        "- `confidence`: 0..1;\n"
        f"- `first_frame`: the first frame (1..{n_frames}) where it becomes visible (for leaving the "
        "workplace: the frame where the cashier stands up / starts to leave); 0 if not visible;\n"
        "- `explanation_ru`: one or two short sentences in Russian describing only what is seen "
        "(no names, no speculation)."
    )


ONSET_QUESTION = (
    "Find the moment the cashier stands up from the cash desk and starts to walk away. "
    "Answer `first_frame`: the frame number where the cashier is standing up / leaving the chair "
    "(0 if the cashier does not leave in these frames), and `left`: true if the cashier leaves the desk."
)

ONSET_SCHEMA = {
    "type": "object",
    "properties": {"left": {"type": "boolean"}, "first_frame": {"type": "integer"}},
    "required": ["left", "first_frame"],
}

GROUND_QUESTION = (
    "Locate the cashier's work desk: the desk surface with the computer/laptop, money counter and "
    "the cash drawer or safe next to it. Return `bbox_2d` = [x1, y1, x2, y2] in coordinates "
    "normalised to 0..1000 of the image width and height. Make the box generous so it includes the "
    "whole desk top and the laptop screen."
)

GROUND_SCHEMA = {
    "type": "object",
    "properties": {"bbox_2d": {"type": "array", "items": {"type": "integer"}}},
    "required": ["bbox_2d"],
}

DESK_OBJECTS = [
    "laptop", "monitor", "keyboard", "mouse", "desk_telephone", "money_counter", "card_reader",
    "printer", "documents_or_papers", "stamp_or_pen", "banknotes", "cash_bundle", "coins", "cash_bag",
    "package_or_envelope", "mobile_phone", "tablet", "cup_or_drink", "food", "handbag", "cosmetics",
    "keys", "other_object",
]

STATE_QUESTION = (
    "Image 1 is the whole room, image 2 an enlarged crop of the cashier's desk at the same moment.\n"
    "- `cashier_at_desk`: is a person sitting or standing at the cashier's working place (the chair "
    "side of the desk with the laptop)? false if that place is empty;\n"
    "- `other_person_behind_desk`: is a SECOND person (not the cashier) on the cashier's side of the "
    "desk, behind the cashier, at the safe or touching the cashier's equipment? People on the "
    "opposite/client side of the desk or just passing by the door do not count;\n"
    "- `objects_on_desk`: every object lying on the desk top, chosen from the list. Note: the "
    "stationary desk telephone with a cord/handset is `desk_telephone`, NOT `mobile_phone`. Use "
    "`mobile_phone` only for a smartphone. Money counting machines are `money_counter`. Name each kind of object once;\n"
    "- `screen`: the laptop/computer screen: `application_window` = lit and showing a program, "
    "document, table, browser or a white/light window; `wallpaper_or_lock_screen` = one full-screen "
    "picture (sky, landscape, blue/dark background, clock, user icon) without program windows; `off` "
    "= black; `uncertain` if you cannot tell; `not_visible` if no screen;\n"
    "- `safe_or_drawer_open`: is a safe door or cash drawer clearly standing open?"
)

STATE_SCHEMA = {
    "type": "object",
    "properties": {
        "cashier_at_desk": {"type": "boolean"},
        "other_person_behind_desk": {"type": "boolean"},
        "objects_on_desk": {"type": "array", "items": {"type": "string", "enum": DESK_OBJECTS}, "maxItems": 14},
        "screen": {"type": "string", "enum": ["application_window", "wallpaper_or_lock_screen", "off",
                                              "uncertain", "not_visible"]},
        "safe_or_drawer_open": {"type": "boolean"},
    },
    "required": ["cashier_at_desk", "other_person_behind_desk", "objects_on_desk", "screen", "safe_or_drawer_open"],
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "observation": {"type": "string"},
        "visible": {"type": "boolean"},
        "confidence": {"type": "number"},
        "first_frame": {"type": "integer"},
        "explanation_ru": {"type": "string"},
    },
    "required": ["observation", "visible", "confidence", "first_frame", "explanation_ru"],
}
