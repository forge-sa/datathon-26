"""Local VLM client (Ollama or any OpenAI-compatible server such as llama-server).

Every call is cached on disk, keyed by model + prompt + image bytes, so re-running the
evaluation after changing fusion thresholds costs no GPU time.
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
from pathlib import Path

import httpx

from .config import CACHE_DIR


class VLMError(RuntimeError):
    pass


class VLMClient:
    def __init__(self, cfg: dict, cache_dir: Path = CACHE_DIR, use_cache: bool = True):
        self.backend = cfg.get("backend", "ollama")
        self.url = cfg.get("url", "http://127.0.0.1:11434").rstrip("/")
        self.temperature = float(cfg.get("temperature", 0.0))
        self.num_ctx = int(cfg.get("num_ctx", 8192))
        self.timeout = float(cfg.get("timeout_sec", 120))
        self.max_tokens = int(cfg.get("max_tokens", 700))   # guards against runaway generation
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- public -------------------------------------------------------------------------
    def chat_json(self, model: str, system: str, frames: list[tuple[str, bytes]], question: str,
                  schema: dict) -> dict:
        """Send labelled frames + question, return parsed JSON matching `schema`.

        frames: list of (label, jpeg bytes), e.g. ("frame 1, t=12.0s", b"...").
        """
        key = self._key(model, system, frames, question, schema)
        cached = self._cache_get(key)
        if cached is not None:
            cached["_cached"] = True
            return cached

        last_err: Exception | None = None
        for attempt in range(2):
            t0 = time.time()
            # retry with a little randomness: at temperature 0 a repetition loop (e.g. the same
            # list item until the token cap) would just repeat
            temp, rep = (self.temperature, 1.05) if attempt == 0 else (max(self.temperature, 0.3), 1.2)
            try:
                text = (self._ollama if self.backend == "ollama" else self._openai)(
                    model, system, frames, question, schema, temp, rep)
                data = _parse_json(text)
                data["_latency"] = round(time.time() - t0, 2)
                LATENCY.update(data["_latency"])
                self._cache_put(key, data)
                return data
            except (ValueError, httpx.HTTPError, KeyError) as e:
                last_err = e
        raise VLMError(f"VLM call failed: {last_err}")

    # -- backends -----------------------------------------------------------------------
    def _ollama(self, model, system, frames, question, schema, temp, rep) -> str:
        legend = "\n".join(f"Image {i + 1}: {label}" for i, (label, _) in enumerate(frames))
        content = f"The {len(frames)} images are video frames in chronological order.\n{legend}\n\n{question}"
        payload = {
            "model": model,
            "stream": False,
            "format": schema,
            "think": False,
            "keep_alive": -1,
            "options": {"temperature": temp, "num_ctx": self.num_ctx,
                        "num_predict": self.max_tokens, "repeat_penalty": rep},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content,
                 "images": [base64.b64encode(b).decode() for _, b in frames]},
            ],
        }
        r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        if r.status_code == 400 and "think" in r.text:
            payload.pop("think")
            r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _openai(self, model, system, frames, question, schema, temp, rep) -> str:
        parts = []
        for label, b in frames:
            parts.append({"type": "text", "text": label})
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b).decode()}})
        parts.append({"type": "text", "text": question})
        payload = {
            "model": model,
            "temperature": temp,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": parts}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
        }
        r = httpx.post(f"{self.url}/v1/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # -- cache --------------------------------------------------------------------------
    def _key(self, model, system, frames, question, schema) -> str:
        h = hashlib.sha1()
        for part in (self.backend, model, system, question, json.dumps(schema, sort_keys=True)):
            h.update(part.encode("utf-8"))
        for label, b in frames:
            h.update(label.encode("utf-8"))
            h.update(hashlib.sha1(b).digest())
        return h.hexdigest()

    def _cache_get(self, key: str) -> dict | None:
        if not self.use_cache:
            return None
        p = self.cache_dir / key[:2] / f"{key}.json"
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                return None
        return None

    def _cache_put(self, key: str, data: dict) -> None:
        p = self.cache_dir / key[:2] / f"{key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


class _Latency:
    """Moving average of seconds per uncached VLM call (used by the time-budget planner)."""

    def __init__(self):
        self.value: float | None = None
        self._lock = threading.Lock()

    def update(self, sec: float) -> None:
        with self._lock:
            self.value = sec if self.value is None else 0.7 * self.value + 0.3 * sec

    def get(self, default: float) -> float:
        return self.value if self.value is not None else default


LATENCY = _Latency()


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in VLM output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def ollama_ready(url: str, model: str) -> tuple[bool, str]:
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=3)
        names = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:  # noqa: BLE001
        return False, f"Ollama is not reachable at {url}: {e}"
    if model not in names and f"{model}:latest" not in names:
        return False, f"model {model} not pulled (have: {', '.join(names) or 'none'})"
    return True, "ok"
