#!/usr/bin/env python3
"""Round 6 / H4 generation: have the *trained* model (Qwopus3.6-27B-v1-preview)
produce fresh <think> traces on the SAME problems used in the natural experiment,
so we can judge its *behavioral* closure and compare to its *training-data* closure.

H4 (incompressible kernel) prediction: Qwopus was SFT'd primarily on
Kassadin88/Claude-Distillation-Dataset (low closure in our audit) + Kimi + Qwen,
then 8B-filtered for stylistic uniformity (which strips overthrow). So if behaviour
inherits training-data closure, Qwopus should reason at LOW-MODERATE closure
(~0.3-1.0), NOT at qwen's 2.08 — despite high-closure qwen being in the mix.

Runs in the `dllm` conda env (transformers 5.12.1 has Qwen3_5ForConditionalGeneration).
Saves outputs/qwopus_traces.jsonl incrementally (resumable). Zero judge calls here;
judging is a separate step that reuses the standard rubric.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qwopus_gen")

# Set QWOPUS_MODEL_DIR to your local snapshot of Jackrong/Qwopus3.6-27B-v1-preview.
MODEL_DIR = os.environ.get("QWOPUS_MODEL_DIR", "models/Qwopus3.6-27B-v1-preview")
OUT = ROOT / "outputs" / "qwopus_traces.jsonl"
N_PROBLEMS = 40
MAX_NEW = 6144
TEMP = 0.6
TOP_P = 0.95


def load_problems(n: int) -> list[dict]:
    """The matched GLM<->DeepSeek problems, with both teachers' closure baselines,
    so Qwopus is evaluated on identical questions to the natural experiment."""
    import run_pilot
    pairs = run_pilot.load_pairs(n)
    ds_ids = {d for d, _ in pairs}
    glm_lk = {g for _, g in pairs} | {d for d, g in pairs if d != g}
    ds_tr = run_pilot.traces_by_uid("deepseek", ds_ids)
    _ = run_pilot.traces_by_uid("glm", glm_lk, file_filter="main")  # warm/validate the pairing side
    probs = []
    for ds_id, glm_id in pairs:
        t = ds_tr.get(ds_id)
        if not t or not (t.problem or "").strip():
            continue
        probs.append({"uid": ds_id, "problem": t.problem.strip(),
                      "ds_uid": ds_id, "glm_uid": glm_id})
        if len(probs) >= n:
            break
    return probs


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    done = set()
    if OUT.exists():
        for line in OUT.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["uid"])
        log.info("resuming; %d already generated", len(done))

    problems = [p for p in load_problems(N_PROBLEMS) if p["uid"] not in done]
    log.info("to generate: %d problems", len(problems))
    if not problems:
        log.info("nothing to do"); return

    log.info("loading tokenizer + model (bf16, device_map=auto across GPUs)...")
    # No trust_remote_code: the qwen3_5 architecture is built into transformers 5.12.x
    # (Qwen3_5ForConditionalGeneration), so we never execute code shipped in the model repo.
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto")
    except Exception as e:  # multimodal class fallback (still no remote code)
        log.warning("AutoModelForCausalLM failed (%s); trying Qwen3_5ForConditionalGeneration", str(e)[:120])
        from transformers import Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map="auto")
    model.train(False)  # inference/eval mode
    log.info("model loaded.")

    with OUT.open("a") as fh:
        for i, p in enumerate(problems):
            t0 = time.time()
            msgs = [{"role": "user", "content": p["problem"][:6000]}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                                         temperature=TEMP, top_p=TOP_P,
                                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            except Exception as e:  # noqa: BLE001
                log.error("gen fail uid=%s: %s", p["uid"], str(e)[:160]); continue
            # the chat template already emitted "<think>\n", so prepend it for clean extraction
            full = "<think>\n" + gen
            closed = "</think>" in full
            think = full.split("</think>")[0].replace("<think>", "").strip() if closed else gen.strip()
            answer = full.split("</think>")[-1].strip() if closed else ""
            rec = {"uid": p["uid"], "dataset": "qwopus_gen", "problem": p["problem"],
                   "reasoning": think, "answer": answer, "closed_think": closed,
                   "reasoning_chars": len(think), "glm_uid": p["glm_uid"], "ds_uid": p["ds_uid"]}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            log.info("[%d/%d] uid=%s think=%dch closed=%s %.1fs",
                     i + 1, len(problems), p["uid"], len(think), closed, time.time() - t0)
    log.info("done -> %s", OUT)


if __name__ == "__main__":
    main()
