# ForteBank Datathon: cashier-violation video audit (local, Windows + RTX 5070)

A local web service. Upload a `.mov` from the overhead camera and it returns either
**"Нарушение не обнаружено"** or **"Обнаружены признаки возможных нарушений"**. For each violation
you get its type, a timecode, 1–3 evidence frames, a description and a quote from the regulation.

Everything runs on this PC: Qwen3-VL-8B through Ollama, RT-DETR person detection, FastAPI.
Nothing is sent to cloud services.

## 1. Setup (once)

```powershell
cd C:\Users\code\Documents\ml
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
ollama pull qwen3-vl:8b-instruct # ~6 GB. NOT "qwen3-vl:8b": that tag is the Thinking model (slow)
# Ollama env (user variables, then restart the Ollama tray app):
setx OLLAMA_NUM_PARALLEL 1; setx OLLAMA_FLASH_ATTENTION 1; setx OLLAMA_KV_CACHE_TYPE q8_0; setx OLLAMA_KEEP_ALIVE -1
```

- **cu128 is required.** The RTX 5070 (sm_120) does not work with older CUDA builds of torch.
- **Person-detector weights** (`PekingU/rtdetr_r50vd`, about 170 MB) download from Hugging Face on the first run.
- **Offline use:** after that first run, set `HF_HUB_OFFLINE=1` and the service needs no internet.

### GPU memory: the one rule that matters

The 8B model at `num_ctx 6144` takes ~11 GB of the 12 GB. If the GPU over-fills, Windows does not
fail: it silently moves GPU memory into system RAM and every VLM call gets 10-30x slower (seen in
testing: a 143 s clip went from 125 s to 1680 s). Symptoms: `nvidia-smi` shows ~12 GB used, 100 %
utilisation but low power draw (~60 W of 250 W). So:

- close LM Studio, Steam, games, Wallpaper Engine and heavy Chrome tabs while the service runs;
- `ollama ps` must say `100% GPU`;
- the person detector runs on the CPU (`perception.device: cpu`) on purpose;
- don't raise `vlm.num_ctx` to 8192 (spills ~18 % of the model to the CPU) or
  `OLLAMA_NUM_PARALLEL` above 1.

## 2. Run the web UI

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1
```

Open **http://127.0.0.1:8000/**.

1. Drop a video and press **Проверить**.
2. The page shows the processing status, then the verdict, the violation cards and the evidence frames. Click a timecode to jump the player to it.
3. To see why the service decided what it did, open **"Технические детали"**. It shows each VLM window's scene text, every candidate with its scores and the verify answers.

`GET /health` reports whether Ollama is reachable and the model is pulled.

### The jury API (also works now, locally)

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/analyze -H "X-API-Key: forte-dev-key" -F "file=@open_data/open/videos/X-1.mov"
curl.exe http://127.0.0.1:8000/api/v1/jobs/<job_id> -H "X-API-Key: forte-dev-key"
```

- **Response:** follows the schema in the brief. The regulation citation is inside `description`. There are extra fields (`type_id`, `confidence`, `rules`), which the brief allows.
- **API key:** change it with `run.ps1 -ApiKey <key>`.
- **Endpoints:** the UI uses `/ui/api/*`, which has no key. Protect or remove it before exposing the service publicly. Tunnel setup is not part of this step.

## 3. How it works

```
.mov → decode once (PyAV, 5 fps JPEGs in memory)
     → RT-DETR person boxes (for evidence frames; person-count rules are optional, off)
     → desk grounding: Qwen3-VL returns the bbox of the cashier's desk once per video
     → WORKPLACE-STATE TIMELINE (main detector), every 1.5 s:
         whole-room image + enlarged desk crop → Qwen3-VL fills a fixed JSON form:
         cashier_at_desk, other_person_behind_desk, objects_on_desk (enumerated list:
         banknotes, cash_bag, mobile_phone, cup, desk_telephone, ...), screen
         (application_window / wallpaper_or_lock_screen / off), safe_or_drawer_open
     → rules over the sequence (app/timeline.py):
         cashier leaves while screen unlocked / cash / phone / open safe stays  → left_workplace_unsecured
         phone, cup, food, bag on the desk while working (≥2 samples)          → personal_items_phone
         safe open ≥4 samples                                                   → safe_left_open
         second person on the cashier side                                      → outsider_in_cash_area
     → open-ended scan: 10 s windows at 1 fps, "any violation from the taxonomy or `other`?"
       (catches types the state rules don't know)
     → fusion: merge same-type hits, score = max(state, scan·w) (+ support)
     → targeted checks: "when exactly does the cashier stand up?", yes/no verify with an
       observation written before the verdict
     → per-type thresholds (stricter for "other", which must be verified) → verdict
     → Russian description template + verbatim quote from config/regulations.yaml
     → evidence frames (with person boxes and a timestamp)
```

Why a state timeline rather than asking "is this a violation?": in this data the difference between
a violation and normal work is mostly the *state* of the workplace (leaving with a lit work screen or
money on the desk vs. leaving after locking up). An 8B VLM answers concrete questions ("which
objects lie on the desk?", "is the screen a program window or a lock screen?") far more reliably than
an abstract judgement, and the rules on top are transparent and tunable.

- **Time budget:** each job has 240 s, counted from the moment of upload. If several videos are queued, the service picks a lighter tier: sparser state samples, then no scan, then no verify. See `tiers` in `config/settings.yaml`.
- **Unseen violation types:** these go to `other`. An `other` hit needs a higher score and must pass the verify pass.

### Files

| Path | What |
|---|---|
| `app/main.py` | FastAPI routes (jury API + UI) |
| `app/jobs.py` | in-process job queue; results in `jobs/<id>/` |
| `app/pipeline.py` | the analysis pipeline above |
| `app/timeline.py` | workplace-state sampling and the state rules |
| `app/video.py` | decoding, frame store, resizing for the VLM |
| `app/vlm.py` | Ollama / OpenAI-compatible client with an on-disk cache (`cache/vlm`) |
| `app/prompts.py` | system prompt, desk-state form, scan / verify / onset questions, JSON schemas |
| `app/perception.py` | RT-DETR person detection, cashier zone, rule candidates |
| `app/fusion.py` | merging, scoring, thresholds (pure functions, replayed by `tune.py`) |
| `config/settings.yaml` | all knobs: model, windows, fps, budget, thresholds |
| `config/taxonomy.yaml` | violation types: definitions for the model, Russian names, linked clauses |
| `config/regulations.yaml` | verified quotes (НБРК No. 231, No. 14, АРРФР No. 53) |
| `config/tuned.yaml` | written by `tools/tune.py --apply`, overrides `settings.yaml` |
| `data/annotations.yaml` | your per-clip types and camera setups (a draft, to be corrected) |
| `tools/` | evaluation, tuning, setups, fine-tuning |

## 4. "Training": how the system gets better

There is no end-to-end model trained on 21 positive clips, because it would overfit and miss new types.
Instead there are three levers, from cheapest to most expensive.

### 4.1 Annotate (do this first)

```powershell
.venv\Scripts\python.exe tools\labels.py          # check the dataset is read correctly
.venv\Scripts\python.exe tools\setups.py --write  # group clips by camera setup → data/annotations.yaml
```

- **Types:** watch each violation clip and fix `type:` in `data/annotations.yaml`. The current types are a draft; X-32 and X-33 are `unknown`.
- **Setups:** check `runs/setups.jpg` and correct `setup:` by eye.

### 4.2 Evaluate

```powershell
.venv\Scripts\python.exe tools\eval.py --run baseline              # all 40 unique clips
.venv\Scripts\python.exe tools\eval.py --run quick --clips X-11 X-23 X-26
.venv\Scripts\python.exe tools\eval.py --run timing --deadline --workers 5   # 5 parallel, real budget
```

- **Per-clip output:** each line shows the label against the prediction.
- **Summary:** recall, specificity, timecode error (mean, ≤1 s, ≤2 s), type accuracy, plus the misses and false alarms by name.
- **Saved files:** every candidate and its features go to `runs/<run>/`.
- **Caching:** VLM calls are cached, so re-running with unchanged prompts and frames takes seconds.

### 4.3 Post-tune thresholds (no GPU, seconds)

```powershell
.venv\Scripts\python.exe tools\tune.py --run baseline            # grid search + leave-one-setup-out CV
.venv\Scripts\python.exe tools\tune.py --run baseline --apply    # write config/tuned.yaml
```

- **What it does:** it replays the fusion step over the cached candidates for many threshold and weight settings. The objective is 1.5·recall + specificity + 0.3·(timecode ≤2 s) + 0.2·type accuracy (`tools/metrics.py`). Recall is weighted higher because a miss is unacceptable. With a 3× weight the search learned to flag every clip, so keep the weight moderate.
- **Which number to trust:** the leave-one-setup-out CV line is the honest estimate for the new-angle test video. The "best on all clips" line is fitted to the same data, so it is optimistic.
- **Applying it:** `--apply` writes `config/tuned.yaml`. The server reloads settings for every job, so no restart is needed. Delete the file to go back to the defaults.

### 4.4 Tune the prompts and taxonomy (needs a new eval run)

Most of the accuracy comes from what the model is asked:

- Edit `definition` / `not_violation` in `config/taxonomy.yaml`. Contrast examples from the no-violation clips help, for example "closes the safe and locks the PC, then leaves = normal".
- Add a new type by adding an entry to the taxonomy, plus its verified clause in `regulations.yaml`. No code changes are needed.
- Try `scan.frame_width` (more detail costs more time), `scan.fps`, `window_sec` / `stride_sec`, and the verify window.
- After each change, run `tools/eval.py --run <new>` and then `tools/tune.py --run <new>`, and compare against the previous run.

### 4.5 Optional: LoRA fine-tuning of Qwen3-VL

Only worth trying after 4.1–4.4, and only keep it if `eval.py` shows a gain.

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-train.txt
.venv\Scripts\python.exe tools\export_sft.py --from-run baseline     # data/sft/train.jsonl, val.jsonl
ollama stop qwen3-vl:8b-instruct                                             # free the GPU
.venv\Scripts\python.exe tools\train_lora.py --model Qwen/Qwen3-VL-4B-Instruct --epochs 2
.venv\Scripts\python.exe tools\train_lora.py --merge runs\lora\Qwen3-VL-4B-Instruct
```

Training details:
- **What it trains:** the scan step only, with the same prompt and frame size the service uses.
- **Training data:** windows from the negative clips are trained to "no events". Windows that contain a labelled onset are trained to that event. Windows after an onset are skipped, because they are ambiguous.
- **Memory:** 4B in bf16 fits in 12 GB with batch 1 and gradient checkpointing. For 8B, add `--load-4bit` (QLoRA).

Serving the fine-tuned model:
1. Convert it to GGUF with llama.cpp. You need the llama.cpp repo for `convert_hf_to_gguf.py`; the winget build has only the binaries.
   ```powershell
   python convert_hf_to_gguf.py runs\lora\Qwen3-VL-4B-Instruct-merged --outtype q8_0 --outfile qwen3vl-forte.gguf
   python convert_hf_to_gguf.py runs\lora\Qwen3-VL-4B-Instruct-merged --mmproj --outfile mmproj-forte.gguf
   llama-server -m qwen3vl-forte.gguf --mmproj mmproj-forte.gguf -ngl 99 -c 8192 --port 8081
   ```
2. In `config/settings.yaml`, set `vlm.backend: openai`, `vlm.url: http://127.0.0.1:8081` and `vlm.model: qwen3vl-forte`.
3. Run `tools/eval.py` again and compare.

Licenses: Qwen3-VL-4B/8B-Instruct are Apache 2.0, so a fine-tuned derivative can be used commercially.

## 5. Models and licenses

| Component | Model | License |
|---|---|---|
| VLM | Qwen3-VL-8B-Instruct (Ollama `qwen3-vl:8b-instruct`, Q4_K_M) | Apache 2.0 |
| Person detection | RT-DETR R50 (`PekingU/rtdetr_r50vd`) | Apache 2.0 |
| Runtime | Ollama (MIT), PyTorch (BSD), FastAPI (MIT), PyAV (BSD) | — |

Do not use Qwen2.5-VL-3B or Qwen2.5-VL-72B (their licenses don't allow it) or Ultralytics YOLO (AGPL). The service does no face recognition.

## 6. Measured results (2026-10-17, 40 unique clips)

| Run | What | Accuracy | Recall | Specificity |
|---|---|---|---|---|
| baseline | open-ended scan + verify + person rules | 50 % | 57 % | 42 % |
| tl4, default settings | state timeline + scan + checks | 55 % | 52 % | 58 % |
| tl4 + tune, fitted on all clips (optimistic) | `config/tuned.yaml` | 65 % | 71 % | 58 % |
| tl4 + tune, **leave-one-setup-out CV** (honest, new angle) | | **60 %** | 57 % | 63 % |

- **Timecodes:** for detected violations, 80 % are within 2 s of the label (tuned, fitted).
- **Speed:** a single clip takes about 35 s on average and 110 s at most (the 143 s clip), uncached. Five uncached clips submitted at once all finished in 110 s or less, against the 300 s limit.
- **What works:** leaving the workplace with cash or an unlocked screen behind (7 of 8 clips), and the safe left open.
- **What doesn't yet:**
  - **Personal items and phone:** the phone is often in the hand rather than on the desk. A "what is the cashier doing" field was tried and was too noisy from single frames.
  - **Outsiders:** hard to separate from the cashier standing at the safe, or from clients.
  - **Cash-handling types:** the draft types for X-1, X-4, X-32 and X-33 are guesses.

**Next lever:** watch the violation clips and correct `type` / `note` in `data/annotations.yaml`. Then iterate on prompts per type (section 4.4), using `tools/eval.py` plus `tools/tune.py` as the check.

## 7. Limitations

- **Labels:** the training labels give only the onset second and no type. The types in `data/annotations.yaml` are a draft.
- **PC lock:** an unlocked PC is reported only as part of "left the workplace". No Kazakh act requires staff to lock the PC on leaving. АРРФР No. 53 п. 71 пп. 9 is only an auto-lock after 30 minutes, so it is not cited.
- **Visibility:** violations that need data the camera can't see (system records, amounts) are never claimed.
- **Throughput:** five videos arriving at once share one GPU, so the service falls back to lighter tiers. Check with `tools/eval.py --deadline --workers 5`.
