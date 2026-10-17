"""Optional: LoRA fine-tuning of Qwen3-VL on the exported scan windows (tools/export_sft.py).

    uv pip install --python .venv\\Scripts\\python.exe -r requirements-train.txt
    # stop Ollama first so the GPU is free:  ollama stop qwen3-vl:8b
    python tools/train_lora.py --model Qwen/Qwen3-VL-4B-Instruct --epochs 2
    python tools/train_lora.py --merge runs/lora/Qwen3-VL-4B-Instruct   # merge adapter -> full HF model

12 GB VRAM: 4B in bf16 with gradient checkpointing and batch 1 fits; 8B needs --load-4bit
(QLoRA, bitsandbytes). With ~21 positive clips this mainly teaches output format and the
camera's look - always compare against the base model with tools/eval.py before switching.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_messages(rec: dict, base: Path, with_answer: bool) -> list[dict]:
    legend = "\n".join(f"Image {i + 1}: {lab}" for i, lab in enumerate(rec["frame_labels"]))
    content = [{"type": "image", "image": str(base / p)} for p in rec["images"]]
    content.append({"type": "text", "text": f"The {len(rec['images'])} images are video frames in chronological "
                                            f"order.\n{legend}\n\n{rec['question']}"})
    msgs = [{"role": "system", "content": [{"type": "text", "text": rec["system"]}]},
            {"role": "user", "content": content}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": rec["target"]}]})
    return msgs


def encode(processor, rec: dict, base: Path):
    full = processor.apply_chat_template(build_messages(rec, base, True), tokenize=True, return_dict=True,
                                         return_tensors="pt")
    prompt = processor.apply_chat_template(build_messages(rec, base, False), tokenize=True, return_dict=True,
                                           return_tensors="pt", add_generation_prompt=True)
    labels = full["input_ids"].clone()
    labels[:, : prompt["input_ids"].shape[1]] = -100     # loss only on the answer
    full["labels"] = labels
    return full


def train(args):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    data_dir = Path(args.data)
    recs = [json.loads(line) for line in open(data_dir / "train.jsonl", encoding="utf-8")]
    val = [json.loads(line) for line in open(data_dir / "val.jsonl", encoding="utf-8")]
    # balance: positives are rare, repeat them so they are ~1/3 of the steps
    pos = [r for r in recs if json.loads(r["target"])["events"]]
    neg = [r for r in recs if not json.loads(r["target"])["events"]]
    if pos:
        recs = neg + pos * max(1, len(neg) // (2 * len(pos)))
    print(f"train windows: {len(recs)} ({len(pos)} unique positive), val: {len(val)}")

    kw = {"torch_dtype": torch.bfloat16, "device_map": "cuda"}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kw)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    lora = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps_total = args.epochs * len(recs) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / max(1, steps_total)))
    out = ROOT / "runs" / "lora" / Path(args.model).name
    out.mkdir(parents=True, exist_ok=True)

    step = 0
    for epoch in range(args.epochs):
        random.Random(epoch).shuffle(recs)
        model.train()
        running = 0.0
        for i, rec in enumerate(recs, 1):
            batch = {k: v.to("cuda") for k, v in encode(processor, rec, data_dir).items()}
            loss = model(**batch).loss / args.accum
            loss.backward()
            running += loss.item()
            if i % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"epoch {epoch + 1} step {step}/{steps_total} loss {running / 10:.4f}", flush=True)
                    running = 0.0
            if args.max_steps and step >= args.max_steps:
                break
        model.eval()
        with torch.no_grad():
            vl = [model(**{k: v.to("cuda") for k, v in encode(processor, r, data_dir).items()}).loss.item()
                  for r in val[: args.val_limit]]
        print(f"epoch {epoch + 1}: val loss {sum(vl) / max(1, len(vl)):.4f}")
        model.save_pretrained(out)
        processor.save_pretrained(out)
        if args.max_steps and step >= args.max_steps:
            break
    print(f"adapter saved to {out}")


def merge(adapter_dir: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    base = AutoModelForImageTextToText.from_pretrained(cfg["base_model_name_or_path"], torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = Path(adapter_dir).with_name(Path(adapter_dir).name + "-merged")
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(cfg["base_model_name_or_path"]).save_pretrained(out)
    print(f"merged model saved to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data" / "sft"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--val-limit", type=int, default=40)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--merge", metavar="ADAPTER_DIR")
    args = ap.parse_args()
    if args.merge:
        merge(args.merge)
    else:
        train(args)


if __name__ == "__main__":
    main()
