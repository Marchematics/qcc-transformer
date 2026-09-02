#!/bin/bash
# Single layer-wise calibration test with memory optimization

set -e

PROJECT_DIR="/home/frankwang122222/zjh/zjh/工作文件/qcc-transformer-next"
MODEL_PATH="$PROJECT_DIR/models/qwen2.5-1.5b"
OUTPUT_DIR="$PROJECT_DIR/artifacts/hf_99/layerwise_test"
RUN_ID="layertest_$(date +%Y%m%d_%H%M%S)"

cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"

# Test: last-quarter layers only, minimal codes, 1024 tokens
LAYERS="last-quarter"
WINDOW=48
CODES=16
STEPS=100
LR=0.02

output_file="$OUTPUT_DIR/${RUN_ID}_layers-${LAYERS}_w${WINDOW}_c${CODES}.pt"
log_file="$OUTPUT_DIR/${RUN_ID}_layers-${LAYERS}_w${WINDOW}_c${CODES}.log"

echo "=== Single layer-wise calibration test ==="
echo "Run ID: $RUN_ID"
echo "Layers: $LAYERS, Window: $WINDOW, Codes: $CODES"
echo "Max tokens: 1024 (to fit in 24GB)"
echo ""

CUDA_VISIBLE_DEVICES=0 \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH="$PROJECT_DIR:$PYTHONPATH" \
python3 benchmarks/calibrate_hf_layerwise.py \
    --model "$MODEL_PATH" \
    --train-file README.md \
    --held-out-file HANDOFF.md \
    --output "$output_file" \
    --calibrate-layers "$LAYERS" \
    --window-size "$WINDOW" \
    --num-codes "$CODES" \
    --steps "$STEPS" \
    --lr "$LR" \
    --max-tokens 1024 \
    --kv-head-policy repeat \
    --run-id "$RUN_ID" \
    --quality-gate 0.99 \
    2>&1 | tee "$log_file"

echo ""
echo "=== Test complete ==="
echo "Result: $output_file"
echo "Log: $log_file"
