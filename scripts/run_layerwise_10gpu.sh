#!/usr/bin/env bash
# Parallel layer-wise calibration sweep. Each configuration owns one GPU.
# This script is diagnostic: it does not create 99-gate evidence by itself.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/qwen2.5-1.5b}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/hf_99/layerwise_10gpu}"
RUN_ID="${RUN_ID:-layerwise10_$(date +%Y%m%d_%H%M%S)}"
MAX_TOKENS="${MAX_TOKENS:-512}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

# gpu:layers:window:codes:steps:lr:cosine_weight
CONFIGS=(
  "0:all:48:16:50:0.01:0.0"
  "1:all:48:16:50:0.01:0.1"
  "2:all:48:16:50:0.01:0.3"
  "3:all:48:16:50:0.01:0.5"
  "4:all:48:16:50:0.01:1.0"
  "5:last-half:48:32:50:0.01:0.3"
  "6:last-half:64:32:50:0.005:0.3"
  "7:last-quarter:48:16:50:0.01:0.3"
  "8:all:32:32:100:0.01:0.3"
  "9:all:48:64:100:0.005:0.3"
)

pids=()
for config in "${CONFIGS[@]}"; do
  IFS=':' read -r gpu layers window codes steps lr cosine_weight <<< "$config"
  stem="${RUN_ID}_gpu${gpu}_layers-${layers}_w${window}_c${codes}_s${steps}_cw${cosine_weight}"
  log_file="$OUTPUT_DIR/${stem}.log"
  output_file="$OUTPUT_DIR/${stem}.pt"
  echo "Launching GPU ${gpu}: layers=${layers} window=${window} codes=${codes} steps=${steps} lr=${lr} cosine_weight=${cosine_weight}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" \
    HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
    PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" \
    python3 benchmarks/calibrate_hf_layerwise.py \
      --model "$MODEL_PATH" \
      --train-file README.md \
      --held-out-file HANDOFF.md \
      --output "$output_file" \
      --calibrate-layers "$layers" \
      --window-size "$window" \
      --num-codes "$codes" \
      --steps "$steps" \
      --lr "$lr" \
      --cosine-weight "$cosine_weight" \
      --max-tokens "$MAX_TOKENS" \
      --kv-head-policy repeat \
      --run-id "$RUN_ID" \
      --quality-gate 0.99
  ) >"$log_file" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done

echo "Sweep ${RUN_ID} complete (status=${status}); results in ${OUTPUT_DIR}"
exit "$status"
