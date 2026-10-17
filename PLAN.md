# ForteBank Datathon: plan for the cashier-violation detection service (Windows / RTX 5070)

## What the brief asks for

You need a service that runs fully locally: a web UI plus an API. It takes a `.mov` from one overhead camera of a cash room and returns a verdict (`violation` / `no_violation`). For each violation it gives a type, a timecode, 1–3 JPEG evidence frames, a short description and a citation of a Kazakh regulation.

- **Scoring is automatic:** the jury runs at least 5 unseen videos through your API. Each must finish within 5 minutes, and the final video is filmed from a new angle.
- **Unseen types:** the test can include violation types that aren't in the training set.
- **Misses vs false alarms:** a missed violation counts as unacceptable, while false alarms should be kept as low as possible.
- **Bans:** cloud AI calls, face recognition and "ready-made solutions" lead to disqualification. Pretrained models are allowed if they are open source with a license that permits commercial use.
- **Other requirements:** only acts of the Republic of Kazakhstan may be cited (НБРК, adilet.zan.kz), a public URL is required, and the deadline is 12:00.

## What the data shows

- **Size and format:** 42 files, all h264 yuv420p, 464×848 portrait, 30 fps, 11–143 s long. There is no rotation metadata and no audio.
- **Duplicates:** X-4 and X-47 are byte-identical, and so are X-5 and X-48. That leaves 40 unique videos: 21 with a violation and 19 without.
  - X-4 is labeled at 57 s and X-47 at 68 s, both as violations, so it's probably one video with two events.
  - X-5 and X-48 are both labeled no-violation.
- **Filename prefixes:** X-1 and X-2 use a Latin `X`; all the others use a Cyrillic `Х`. Normalize with `name.replace("Х", "X")` before joining videos with labels.
- **Labels:** only "violation at second N", with no type, so you define the types.
  - The time is free text ("22 секунда", "1минута 8 секунда"); parse it with a regex.
  - X-6, 7, 9, 36, 41 and 42 are missing, probably held out for the test.
- **Camera setups:** training already has at least 4 different angles or room layouts, so you can test robustness to a new angle with what you have.
- **What separates violations from normal clips is the state of things, not the action.**
  - In X-23 and X-29 (no violation) the cashier closes the safe, locks the laptop to a blue lock screen, and then leaves.
  - In X-26 and X-28 (violations) she leaves with cash on the desk or with the work app open.
- **Timecode convention:** for "left the workplace" violations, the label marks when the cashier stands up or heads for the door.

### Violation groups (a reading of frames around each label; confirm by watching the full clips)

| Group | Clips | Regulation (checked in the legal text) |
|---|---|---|
| Left the workplace with valuables unsecured: cash or a bag on the desk, safe open, PC unlocked | X-11, 15, 22, 24, 26, 28 | Resolution No. 231 p.20 (only the current client's cash on the desk; the rest in lockable storage), p.22 (don't leave before finishing with the client) |
| Outsider in the cash room or on the cashier's side of the desk | X-18, 30, 34, 38 | No. 231 p.75 (safe-room access); No. 14 p.16(1) (cash room has a lockable door) |
| Personal items, phone or food at the workplace | X-20, 40, 46 (maybe X-2, 44) | No. 231 p.12 (personal items kept outside the cash unit) |
| Cash handling with a client: cash left on the desk, recounting, ID check | X-1, X-4/47, (X-2) | No. 231 p.20, p.31 (ID check, recount in front of client), p.35 |
| Collector bag accepted without checking it | X-17 | No. 231 p.24 (check the bag is intact and sealed) |
| Safe left open | X-13, (X-20) | No. 231 p.20 |
| Still unclear | X-32, X-33 | to be determined |

**Sources:**
- Resolution No. 231 of 29.11.2019, amended up to 10.11.2025: [zakon.uchet.kz/V1900019680](https://zakon.uchet.kz/rus/history/V1900019680/16.12.2020)
- Resolution No. 14 of 24.02.2020, amended 24.12.2024: [zakon.uchet.kz/V2000020075](https://zakon.uchet.kz/rus/docs/V2000020075)

For the PC-lock violation we still need an information-security regulation. Don't cite one until it's found and checked, because invented citations lose points.

## Target machine and runtime

**Machine (checked 2026-10-17):**
- **System:** Windows 10 Pro 19045, i7-14700K (20 cores / 28 threads), 32 GB RAM, about 670 GB free disk.
- **GPU:** RTX 5070 with 12 GB VRAM (Blackwell, sm_120). Driver 616.64, CUDA 13.4. About 1.3 GB of VRAM is used by the desktop.
- **Already installed:** Python 3.12 (`py`) and `uv`, ffmpeg 8.1.1, Ollama 0.12.10, llama.cpp (`llama-server`), Docker Desktop (WSL2), WSL Ubuntu, git.
- **Missing:** cloudflared.

| Item | Choice |
|---|---|
| OS | Windows 10 native. No WSL or Docker in the dev loop. |
| Python | 3.12 via `uv venv --python 3.12` in `ml\.venv` |
| PyTorch | **cu128 or newer wheels** (`--index-url https://download.pytorch.org/whl/cu128`). Older CUDA builds have no sm_120 kernels and fail on the RTX 5070. |
| VLM server | **Ollama**: upgrade to the latest version so `qwen3-vl` is supported. `llama-server` is the fallback. |
| Video I/O | PyAV or `opencv-python`. Avoid decord, which has poor Windows wheels. ffmpeg on PATH handles clipping and re-encoding. |
| Detectors | PyTorch + HF `transformers` on GPU (RT-DETR / OWLv2 / Grounding DINO). rtmlib pose runs on CPU via onnxruntime, which is fine with 20 cores. |
| API | FastAPI + uvicorn in **one process**. Jobs go through an in-process queue: a thread pool for decode and CPU work, plus a single GPU worker. Windows has no `fork`, so avoid multi-worker uvicorn. |
| Public URL | `winget install Cloudflare.cloudflared`, then a named Cloudflare Tunnel to `localhost:8000`, run as a Windows service. |
| Encoding | Set `PYTHONUTF8=1`. Handle Cyrillic paths only through Python `pathlib`, because PowerShell 5.1 mangles them in inline commands. |
| Portability deliverable | A `docker-compose.yml` with the `ollama/ollama` service plus the app. It gets the GPU through Docker Desktop's WSL2 backend, sets `HF_HUB_OFFLINE=1`, and bakes the weights in. Build and smoke-test it once in P3; it isn't the dev path. |
| Judging-day hygiene | Turn off sleep (`powercfg /change standby-timeout-ac 0`) and pause Windows Update. Close Chrome, Wallpaper Engine, SignalRGB and Steam to free VRAM. Start everything from `start.ps1`. |

**VRAM budget (12 GB):**

| Component | VRAM |
|---|---|
| Desktop | about 1.3 GB |
| Qwen3-VL-8B Q4 with 8k context | about 7 GB |
| RT-DETR + OWLv2 | about 1.5–2 GB |
| Headroom | about 1.5 GB |

Ollama environment: `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_CONTEXT_LENGTH=8192`, `OLLAMA_NUM_PARALLEL=2`.

## Qwen options (all must allow commercial use)

| Model | License | Q4 size | Fits 12 GB with detectors | Role |
|---|---|---|---|---|
| **Qwen3-VL-8B-Instruct** (`ollama pull qwen3-vl:8b`) | Apache 2.0 | about 6 GB | yes | **Primary**: scan, verify, describe |
| Qwen3-VL-4B-Instruct (`qwen3-vl:4b`) | Apache 2.0 | about 3.3 GB | yes | Fast fallback when the queue is long or time is short |
| Qwen2.5-VL-7B-Instruct (`qwen2.5vl:7b`) | Apache 2.0 | about 6 GB | yes | A/B baseline only; weaker temporal grounding |
| Qwen3-VL-30B-A3B-Instruct | Apache 2.0 | about 19 GB | no; needs llama.cpp `--n-cpu-moe` offload into the 32 GB of RAM | Offline helper for drafting annotations. Too slow to serve. |
| Qwen3-VL-*-Thinking | Apache 2.0 | — | — | Don't use for serving: the long reasoning output eats the time budget |
| Qwen2.5-VL-3B / 72B | Qwen Research / Qwen license | — | — | **Do not use**: license problem |
| Native-video Qwen3-VL via `transformers` (4B in bf16 ≈ 9 GB) | Apache 2.0 | — | competes with the detectors for VRAM | Offline experiments only. It takes real video with timestamps but is slower than Ollama on Windows. vLLM doesn't run natively on Windows. |

Check each checkpoint's license on its HF model card before use, and record it in the submission docs.

**P0 smoke test:**
1. Send 10 timestamped frames to `qwen3-vl:8b` through `/api/chat` with a JSON-schema `format`.
2. Measure prefill time, generation time and VRAM use.
3. If multi-image input or structured output misbehaves in Ollama, switch to `llama-server -m Qwen3-VL-8B-Instruct-Q4_K_M.gguf --mmproj <mmproj>.gguf` behind its OpenAI-compatible endpoint. The client code stays the same.

## Architecture

A model trained end-to-end on 21 positive clips would overfit and miss the new types. Instead, the service extracts what happens in the scene, applies explicit rules, and uses a local VLM (now on GPU) as the core scanner, checker and describer.

```
.mov ──► decode once (PyAV)
          ├─ 5 fps @ 640px  ─► perception (GPU): person det + ByteTrack, pose (CPU),
          │                    OWLv2 prompts: banknotes / safe door / cash bag / phone / laptop screen
          │                    ─► auto zones (desk, cashier side, client side, safe, door) from detections
          │                    ─► per-second signal timeline (people per zone, cashier present,
          │                       cash on desk, safe open, screen state, phone/food)
          │                    ─► rule engine ─► rule candidates (type, t, score)
          └─ 2 fps JPEGs to job dir (VLM input, evidence frames, UI thumbnails)

rule candidates + uniform scan windows ─► VLM (Qwen3-VL-8B, Ollama, JSON schema)
   ─► parse + validate ─► temporal merge/NMS ─► timecode refinement ─► verify pass
   ─► regulations.yaml lookup ─► Russian description template ─► evidence frames (boxes) ─► JSON
```

- **Zones** come from detected objects, not fixed pixels, so a new angle still works.
- **Roles** are assigned by position: the cashier is whoever stays in the cashier zone. No faces are involved.

### Video slicing

1. **Decode once.** Read the file with PyAV a single time.
   - Detectors get a 5 fps stream at 640 px on the long side, processed on the fly rather than held in RAM.
   - Write 2 fps JPEGs at native 464×848, quality 85, to `jobs/<id>/frames/`. A 143 s clip gives about 290 files of roughly 40 KB each.
2. **Scan windows** cover the whole video, so new types get caught.
   - Each window is **10 s long with a 5 s stride**, sampled at **1 fps** (10 frames).
   - Frames are resized to about 336×616, roughly 200 visual tokens each (Qwen3-VL uses 32 px merged patches). With the prompt, that's about 2.5k tokens per call.
   - The 143 s clip needs about 28 calls; a typical 15–30 s clip needs 3–6.
   - Put a text tag such as `t=12.0s` before each image, so the model can refer to absolute times.
3. **Candidate windows.** Each rule candidate gets an extra window: ±4 s around t at **2 fps** (16 frames). When the relevant zone is known, add a crop of it.
4. **Refinement window.** For each event that survives, take ±3 s from the source video at **5 fps** and ask: "In which frame does X first become true?"
   - Snap to a perception signal when one exists, such as the cashier's hip keypoints leaving the desk zone, or cash appearing on the desk.
   - Follow the label convention: for "left the workplace," the timecode is when the cashier stands up.
5. **Budget control.** Each job has a 240 s deadline, checked before every stage. When time runs short, degrade in this order:
   1. Scan at 0.5 fps or with a 10 s stride.
   2. Switch the scan to `qwen3-vl:4b`.
   3. Skip the verify pass.
   4. Return the rules-only result.

   Five parallel uploads share one GPU, so the queue length at submission picks the starting tier.

### Turning VLM output into violations

- **Taxonomy.** The prompt lists the type ids from `regulations.yaml`, plus `other`:
  - `left_workplace_unsecured`, `outsider_in_cash_area`, `personal_items_phone`, `cash_left_on_desk`, `collector_bag_unchecked`, `safe_left_open`.
  - Each id comes with a one-line visual definition and a contrast example from the no-violation clips. For instance, in X-23 and X-29 the safe is closed and the screen is locked, so leaving is not a violation.
- **Structured output.** Use Ollama `format` with a JSON schema, at temperature 0:
  ```json
  {"events":[{"type_id":"...","start_sec":0,"end_sec":0,"confidence":0.0,
              "what_is_visible":"...", "evidence_t":[0.0]}]}
  ```
  - Validate with pydantic. On a parse failure, retry once, then drop the result.
  - Clamp times to the window and snap them to the nearest sampled frame.
- **Fusion.**
  - Merge events of the same type whose intervals overlap or fall within 5 s of each other; this removes duplicates from overlapping windows.
  - An event's score is the higher of the VLM confidence and the rule score, with a bonus when both agree.
  - Taxonomy types pass at a low threshold, tuned for recall.
  - `other` needs a higher threshold and must pass the verify pass. This catches unseen types without flooding false alarms.
- **Verify pass.** Each kept event's window is asked a yes/no question: "Is <definition> visible? Answer with frames."
  - The frames are shifted by +0.5 s, so the check isn't just an echo of the scan.
  - An event that fails is dropped, unless it is rule-backed with a high score.
- **Verdict.** `violation` if any event survives; otherwise `no_violation`.
- **Description.** A Russian template fills in:
  - what is visible (from `what_is_visible`, trimmed);
  - why it is a violation;
  - the citation (act, point and a short quote) from the YAML. The model never generates citations.
- **Evidence frames.** 1–3 frames: the refined onset plus one before and one after, with zone and object boxes drawn.

### Model licenses (all must allow commercial use)
- **Detection:** RT-DETR, D-FINE or RF-DETR (Apache 2.0). Avoid Ultralytics YOLO, which is AGPL.
- **Tracking:** ByteTrack (MIT).
- **Pose:** RTMPose via rtmlib (Apache 2.0).
- **Open-vocabulary detection:** OWLv2 or Grounding DINO (Apache 2.0), or Florence-2 (MIT).
- **VLM:** see the Qwen table above.

### API details that are easy to get wrong
- **Response schema:** match it exactly. Put the regulation citation inside `description`, because the schema has no field for it. Extra fields such as `rule` or `confidence` are fine to add.
- **API key:** reject requests without a valid `X-API-Key`.
- **Concurrency:**
  - Run one uvicorn process.
  - A single GPU worker serializes VLM calls; Ollama `NUM_PARALLEL=2` lets two overlap.
  - The budget tiers keep the 5th of 5 simultaneous uploads under 5 minutes.
- **Job state:** keep status in an in-memory dict and mirror it to `jobs/<id>/result.json`, so it survives a restart.
- **Offline weights:** pre-download all weights and set `HF_HUB_OFFLINE=1`, so the service can move into the bank's network.

## Phases (in priority order)

**P0 – foundation and risk reduction**
1. **Environment.**
   - Run `uv venv --python 3.12`, install torch cu128, and check that `torch.cuda.get_device_capability()` returns `(12, 0)`.
   - Upgrade Ollama, then `ollama pull qwen3-vl:8b` and `qwen3-vl:4b`.
   - Run the VLM smoke test and timing measurement.
2. **API skeleton.** A FastAPI app with the exact schema, the API-key check, a job queue and a placeholder analyzer. Expose it now through cloudflared.
3. **Annotation.**
   - Write `tools/label_loader.py` to normalize X/Х and parse label times.
   - Watch all 21 violation clips and write the annotation file: type, start/end, notes, camera setup.
   - Optionally, use Qwen3-VL-30B-A3B offline to pre-draft per-clip descriptions.
4. **Regulations.** Build `regulations.yaml` from verified clauses only.
5. **Evaluation.**
   - Write an eval script that scores the verdict, timecode error (±1 s / ±2 s) and type.
   - Validate by holding out one camera setup at a time.
   - Run all 40 unique clips through the real API.

**P1 – core detection**
- The VLM scan, output parsing and fusion.
- Perception plus rules for the three biggest groups: leaving the workplace with valuables unsecured, outsiders, and personal items or phone. Together they cover about 13–15 of the 21 positives.

**P2 – remaining groups and refinement**
1. Rules for cash handling, the collector bag and the safe.
2. The verify pass and timecode refinement.
3. Evidence frames with boxes drawn.

**P3 – robustness and delivery**
1. Film your own clips from new angles (the FAQ allows this) to stand in for the new-angle test.
2. Tune thresholds for high recall with few false alarms.
3. **Load-test 5 parallel uploads.** All must finish under 5 minutes on this GPU.
4. Write `start.ps1` (Ollama with env vars, the app, cloudflared, a health check) and the docker-compose portability build.
5. Polish the Russian UI:
   - a video player that jumps to the timecode;
   - a note that the result is a hint for a human reviewer;
   - processing status.
6. Write the submission material from section 8 of the brief:
   - architecture, sources, violation types and limitations;
   - every model with its license;
   - the pitch.

## Main risks
- **New angle:** zones must come from detected objects, not fixed coordinates, and testing must hold out whole camera setups.
- **Unseen violation types:** the VLM scan catches them, but `other` is the main source of false alarms. It needs a higher threshold and a verify pass.
- **GPU contention with 5 parallel videos:** measure per-call latency in P0 and rely on the degradation tiers so every job finishes under 5 minutes.
- **Blackwell (sm_120) compatibility:** pin torch cu128 or newer, and use onnxruntime on CPU only unless its GPU build has been verified.
- **The home PC is the production server:** power loss, sleep, updates or a dropped tunnel all mean failed videos. Run cloudflared as a service, disable sleep, use a UPS if one is available, and keep a health-check script running.
- **Ollama version drift:** pin the Ollama version and the model digests once they're validated.
- **Invented citations:** cite only clauses quoted from the official text.

## Open questions
1. **Deadline and team.** Which day is the 12:00 deadline, and how many people are working on this? That decides how far past P1 to go. Hosting is settled: this PC with the RTX 5070 serves the demo through a Cloudflare Tunnel.
