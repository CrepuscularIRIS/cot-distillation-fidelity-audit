#!/usr/bin/env python3
"""Round 6 / H4 generation, model-agnostic. Have any local HF causal-LM produce
fresh <think> traces on the SAME problems used in the natural experiment, so its
behavioral closure (CCR) is directly comparable to the dataset closures we audited.

Controlled inheritance test (the killer H4):
  Qwen3.5-9B-GLM5.1-Distill   (trained on GLM, native CCR 1.67)  vs
  Qwen3.5-9B-DeepSeek-V4-Flash(trained on DeepSeek, native CCR 0.60)
  -> identical base + scale, only the teacher's distillation data differs.
Cross-scale test: Qwopus3.5-9B-v3 vs Qwopus3.6-27B-v1-preview.

Usage (in the `dllm` conda env; transformers 5.12.x has Qwen3_5ForConditionalGeneration):
  python scripts/gen_model_traces.py --model-dir models/Qwen3.5-9B-GLM5.1-Distill-v1 \
         --tag glm9b --out outputs/h4_glm9b_traces.jsonl
Saves incrementally (resumable). No judge calls here; judging is a separate step.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gen_model")


def load_problems(n: int) -> list[dict]:
    """The matched GLM<->DeepSeek problems (same as the natural experiment)."""
    import run_pilot
    pairs = run_pilot.load_pairs(n)
    ds_ids = {d for d, _ in pairs}
    ds_tr = run_pilot.traces_by_uid("deepseek", ds_ids)
    probs = []
    for ds_id, glm_id in pairs:
        t = ds_tr.get(ds_id)
        if t and (t.problem or "").strip():
            probs.append({"uid": ds_id, "problem": t.problem.strip(), "glm_uid": glm_id})
        if len(probs) >= n:
            break
    return probs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True, help="dataset label for the output rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max-new", type=int, default=6144)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.out)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uid"])
        log.info("resuming %s; %d already generated", out.name, len(done))
    problems = [p for p in load_problems(args.n) if p["uid"] not in done]
    if not problems:
        log.info("nothing to do for %s", args.tag); return
    log.info("[%s] generating %d traces from %s", args.tag, len(problems), args.model_dir)

    # No trust_remote_code: qwen3_5 is built into transformers; never run repo code.
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
    model.train(False)
    log.info("[%s] model loaded", args.tag)

    with out.open("a") as fh:
        for i, p in enumerate(problems):
            t0 = time.time()
            msgs = [{"role": "user", "content": p["problem"][:6000]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            primes_think = prompt.rstrip().endswith("<think>")
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            try:
                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new, do_sample=True,
                        temperature=args.temperature, top_p=args.top_p,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen = tok.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            except Exception as e:  # noqa: BLE001
                log.error("[%s] gen fail uid=%s: %s", args.tag, p["uid"], str(e)[:160]); continue
            full = ("<think>\n" + gen) if primes_think else gen
            closed = "</think>" in full
            if closed:
                think = full.split("</think>")[0].replace("<think>", "").strip()
                answer = full.split("</think>")[-1].strip()
            else:
                think, answer = full.strip(), ""
            rec = {"uid": p["uid"], "dataset": args.tag, "problem": p["problem"],
                   "reasoning": think, "answer": answer, "closed_think": closed,
                   "reasoning_chars": len(think), "glm_uid": p["glm_uid"], "model_dir": args.model_dir}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            log.info("[%s %d/%d] uid=%s think=%dch closed=%s %.1fs",
                     args.tag, i + 1, len(problems), p["uid"], len(think), closed, time.time() - t0)
    log.info("[%s] done -> %s", args.tag, out)


if __name__ == "__main__":
    main()
