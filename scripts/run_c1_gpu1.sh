#!/bin/bash
# C1 Closure Inheritance — GPU 1 group: ds9b, gemini9b
set -euo pipefail
cd /home/lingxufeng/huggingface/cot-distillation-fidelity-audit
MODELS_DIR=/data/huggingface/models
LOG=outputs/c1_trace_gen.log
export CUDA_VISIBLE_DEVICES=1

run_model() {
    local dir="$1" tag="$2" out="$3"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 40 ]; then
        echo "[c1-gpu1] SKIP $tag (already has $(wc -l < "$out") traces)" | tee -a "$LOG"
        return 0
    fi
    echo "[c1-gpu1] START $tag from $dir" | tee -a "$LOG"
    conda run -n dllm python scripts/gen_model_traces.py \
        --model-dir "$dir" --tag "$tag" --out "$out" \
        --n 40 --max-new 6144 --temperature 0.6 2>&1 | tee -a "$LOG"
    echo "[c1-gpu1] DONE $tag -> $out ($(wc -l < "$out") traces)" | tee -a "$LOG"
}

echo "=== C1 GPU1 started $(date) ===" | tee -a "$LOG"
run_model "$MODELS_DIR/Qwen3.5-9B-DeepSeek-V4-Flash" "ds9b" "outputs/h4_ds9b_traces.jsonl"
run_model "$MODELS_DIR/Qwen3.5-9B-Gemini-3.1-Pro-Reasoning-Distill" "gemini9b" "outputs/h4_gemini9b_traces.jsonl"
echo "=== C1 GPU1 finished $(date) ===" | tee -a "$LOG"
