"""Paths, environment and YAML config loading."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", ROOT / "config"))
JOBS_DIR = Path(os.environ.get("JOBS_DIR", ROOT / "jobs"))
CACHE_DIR = Path(os.environ.get("VLM_CACHE_DIR", ROOT / "cache" / "vlm"))
STATIC_DIR = Path(__file__).resolve().parent / "static"

API_KEY = os.environ.get("API_KEY", "forte-dev-key")


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_settings() -> dict:
    """settings.yaml, with config/tuned.yaml (written by tools/tune.py --apply) merged on top."""
    s = load_yaml("settings.yaml")
    if (CONFIG_DIR / "tuned.yaml").exists():
        _deep_merge(s, load_yaml("tuned.yaml"))
    return s


def _deep_merge(base: dict, over: dict) -> None:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def load_taxonomy() -> dict[str, dict]:
    return {t["id"]: t for t in load_yaml("taxonomy.yaml")["types"]}


def load_regulations() -> dict:
    data = load_yaml("regulations.yaml")
    sources = {s["id"]: s for s in data.get("sources", [])}
    clauses = {c["id"]: c for c in data.get("clauses", [])}
    return {"sources": sources, "clauses": clauses}
