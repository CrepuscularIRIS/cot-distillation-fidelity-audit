#!/bin/bash
# C1 Closure Inheritance — GPU 0 group: glm9b, opus9b, base9b
set -euo pipefail
cd /home/lingxufeng/huggingface/cot-distillation-fidelity-audit
MODELS_DIR=/data/huggingface/models
LOG=outputs/c1_trace_gen.log
export CUDA_VISIBLE_DEVICES=0

run_model() {
    local dir="$1" tag="$2" out="$3"
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -ge 40 ]; then
        echo "[c1-gpu0] SKIP $tag (already has $(wc -l < "$out") traces)" | tee -a "$LOG"
        return 0
    fi
    echo "[c1-gpu0] START $tag from $dir" | tee -a "$LOG"
    conda run -n dllm python scripts/gen_model_traces.py \
        --model-dir "$dir" --tag "$tag" --out "$out" \
        --n 40 --max-new 6144 --temperature 0.6 2>&1 | tee -a "$LOG"
    echo "[c1-gpu0] DONE $tag -> $out ($(wc -l < "$out") traces)" | tee -a "$LOG"
}

echo "=== C1 GPU0 started $(date) ===" | tee -a "$LOG"
run_model "$MODELS_DIR/Qwen3.5-9B-GLM5.1-Distill-v1" "glm9b" "outputs/h4_glm9b_traces.jsonl"
run_model "$MODELS_DIR/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled" "opus9b" "outputs/h4_opus9b_traces.jsonl"
run_model "$MODELS_DIR/Qwen3.5-9B" "base9b" "outputs/h4_base9b_traces.jsonl"
echo "=== C1 GPU0 finished $(date) ===" | tee -a "$LOG"
