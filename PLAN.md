# ForteBank Datathon: plan for the cashier-violation detection service

## What the brief asks for

You need a service that runs fully locally: a web UI plus an API. It takes a `.mov` from one overhead camera of a cash room and returns a verdict (`violation` / `no_violation`). For each violation it gives a type, a timecode, 1–3 JPEG evidence frames, a short description and a citation of a Kazakh regulation.

- **Scoring is automatic:** the jury runs at least 5 unseen videos through your API. Each must finish within 5 minutes, and the final video is filmed from a new angle.
- **Unseen types:** the test can include violation types that aren't in the training set.
- **Misses vs false alarms:** a missed violation counts as unacceptable, while false alarms should be kept as low as possible.
- **Bans:** cloud AI calls, face recognition and "ready-made solutions" lead to disqualification.

## What the data shows

- **Size and format:** 42 files, all 464×848 portrait at 30 fps, 11–143 s long.
- **Duplicates:** X-4 and X-47 are byte-identical, and so are X-5 and X-48. That leaves 40 unique videos: 21 with a violation and 19 without. X-4 is labeled 57 s and X-47 68 s, so it's probably one video with two events.
- **Labels:** only "violation at second N", with no type. You define the types.
- **Camera setups:** training already has at least 4 different angles or room layouts, so you can test robustness to a new angle with what you have.
- **What separates violations from normal clips is the state of things, not the action.** In X-23 and X-29 (no violation) the cashier closes the safe, locks the laptop to a blue lock screen, and then leaves. In X-26 and X-28 (violations) she leaves with cash on the desk or with the work app open.
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

## Architecture

A model trained end-to-end on 21 positive clips would overfit and miss the new types. Instead, the service extracts what happens in the scene, applies explicit rules, and uses a local vision-language model to check and describe the results.

```
.mov → frame sampling (~5 fps)
     → perception: people detection + tracking + pose, objects (phone, cup, handbag, laptop),
       open-vocabulary detection ("banknotes", "safe", "cash bag"), state checks
       (safe open/closed, screen locked/active, door)
     → automatic zones (desk / cashier side / client side / safe / door) from detected objects,
       not fixed pixels, so a new angle still works
     → roles by position (cashier = whoever stays in the cashier zone); no faces involved
     → timeline of events → rule engine → candidate violations with timecodes
     → local VLM check on a short clip around each candidate (yes/no + what is visible)
       + one general checklist pass to catch types the rules don't cover
     → regulations file (YAML: type → act, point, quoted text) → description from a template
       plus what the model confirmed
     → 1–3 evidence frames (at the timecode plus context, with boxes drawn)
```

**Model choices.** All need a license that allows commercial use, since ForteBank may use the solution commercially:
- **Detection:** RT-DETR, D-FINE or RF-DETR (Apache 2.0). Avoid Ultralytics YOLO, which is AGPL.
- **Tracking:** ByteTrack (MIT).
- **Pose:** RTMPose via rtmlib (Apache 2.0).
- **Open-vocabulary detection:** OWLv2 or Grounding DINO (Apache 2.0), or Florence-2 (MIT).
- **VLM:** Qwen2.5-VL-7B or a Qwen3-VL checkpoint (Apache 2.0). Don't use Qwen2.5-VL-3B: its research license doesn't allow commercial use. Check the license of each checkpoint before using it.

**Hardware.** The dev laptop has no discrete GPU (i7-13620H, 30 GB RAM, integrated graphics).
- The perception part runs fine on CPU with ONNX/OpenVINO.
- A VLM on CPU (llama.cpp, quantized model of about 4B parameters) would take most of the 5-minute budget.
- So the VLM must be optional: if time runs short, return the rule-based result without it.

**API details that are easy to get wrong:**
- **Response schema:** match it exactly. Put the regulation citation inside `description`, because the schema has no field for it; extra fields like `rule` or `confidence` are fine to add.
- **Time limit when videos arrive together:** if the jury submits 5 videos at once and they run one after another, the last ones will exceed 5 minutes. Use parallel workers, or cut the per-video time budget when the queue is long.
- **Offline weights:** bake them into a Docker image and set `HF_HUB_OFFLINE=1`, so the service can move into the bank's network.

## Phases (in priority order)

**P0 – foundation and risk reduction**
1. Set up the environment in Docker; NixOS and Python 3.14 will cause trouble with PyTorch wheels otherwise.
2. Stand up a FastAPI skeleton with the exact schema, API-key check, job queue and a placeholder analyzer, and put it on a public URL now.
3. Watch all 21 violation clips and write an annotation file: type, start/end, notes, camera setup.
4. Build the regulations YAML from verified clauses only.
5. Write an evaluation script that scores the verdict, timecode error (±1 s / ±2 s) and type. Validate by holding out one camera setup at a time.

**P1 – core detection.** Perception pipeline plus the three biggest groups (unsecured workplace on leaving, outsider, personal items/phone). Together they cover about 13–15 of the 21 positives.

**P2 – remaining groups and the model.**
1. Rules for cash handling, the collector bag and the safe.
2. VLM check and descriptions.
3. Evidence frames with boxes drawn.

**P3 – robustness and delivery.**
1. Film your own clips from new angles (the FAQ allows this) to stand in for the new-angle test.
2. Tune thresholds for high recall with few false alarms.
3. Load-test 5 parallel uploads against the 5-minute limit.
4. Polish the Russian UI: video player that jumps to the timecode, a note that the result is a hint for a human reviewer, and processing status.
5. Write the submission material from section 8 of the brief: architecture, sources, violation types, limitations; plus the pitch.

## Main risks
- **New angle:** zones must come from detected objects, not fixed coordinates, and testing must hold out whole camera setups.
- **Unseen violation types:** the VLM checklist pass can catch them, but it is the main source of false alarms, so it needs a confidence threshold.
- **Timing and concurrency:** each video needs its own internal time budget so it always finishes under 5 minutes.
- **Invented citations:** cite only clauses quoted from the official text.

## Open questions
1. **Hosting for judging.** Can the team rent a GPU server, or does a teammate have a GPU? Without one, the VLM step is cut down or dropped, and the public URL would have to be the dev laptop behind a Cloudflare Tunnel.
2. **Deadline and team.** Which day is the 12:00 deadline, and how many people are working on this? That decides how far past P1 to go.
