#!/bin/bash
# C1 Closure Inheritance — sequential trace generation for all student models.
# Each model: ~18GB bf16, fits on single 4090D. ~40min per model, 40 problems.
set -euo pipefail

cd /home/lingxufeng/huggingface/cot-distillation-fidelity-audit
MODELS_DIR=/home/lingxufeng/huggingface/models
LOG=outputs/c1_trace_gen.log
export CUDA_VISIBLE_DEVICES=0

echo "=== C1 trace generation started $(date) ===" | tee -a "$LOG"

run_model() {
    local dir="$1" tag="$2" out="$3"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 40 ]; then
        echo "[c1] SKIP $tag (already has $(wc -l < "$out") traces)" | tee -a "$LOG"
        return 0
    fi
    echo "[c1] START $tag from $dir" | tee -a "$LOG"
    conda run -n dllm python scripts/gen_model_traces.py \
        --model-dir "$dir" --tag "$tag" --out "$out" \
        --n 40 --max-new 6144 --temperature 0.6 2>&1 | tee -a "$LOG"
    echo "[c1] DONE $tag -> $out ($(wc -l < "$out") traces)" | tee -a "$LOG"
}

# Phase 1: Core pair (GLM high-CCR vs DeepSeek low-CCR)
run_model "$MODELS_DIR/Qwen3.5-9B-GLM5.1-Distill-v1" "glm9b" "outputs/h4_glm9b_traces.jsonl"
run_model "$MODELS_DIR/Qwen3.5-9B-DeepSeek-V4-Flash" "ds9b" "outputs/h4_ds9b_traces.jsonl"

# Phase 2: Extended comparison
run_model "$MODELS_DIR/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled" "opus9b" "outputs/h4_opus9b_traces.jsonl"

# Phase 3: These need completed downloads — check before running
if [ -f "$MODELS_DIR/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill/model.safetensors-00001-of-00004.safetensors" ]; then
    run_model "$MODELS_DIR/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill" "gemini9b" "outputs/h4_gemini9b_traces.jsonl"
else
    echo "[c1] SKIP gemini9b (download incomplete)" | tee -a "$LOG"
fi

if [ -f "$MODELS_DIR/Qwen3.5-9B/model.safetensors-00001-of-00004.safetensors" ]; then
    run_model "$MODELS_DIR/Qwen3.5-9B" "base9b" "outputs/h4_base9b_traces.jsonl"
else
    echo "[c1] SKIP base9b (download incomplete)" | tee -a "$LOG"
fi

echo "=== C1 trace generation finished $(date) ===" | tee -a "$LOG"
