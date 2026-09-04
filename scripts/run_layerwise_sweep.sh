#!/bin/bash
# Layer-wise calibration sweep to reach >=0.99 held-out fidelity
# Priority: later layers first, as they're closer to logits

set -e

PROJECT_DIR="${PROJECT_DIR:-/mnt/workspace/qcc-transformer}"
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/phi-4-mini-instruct-ms}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/artifacts/hf_99/layerwise_sweep}"
RUN_ID="${RUN_ID:-layerwise_$(date +%Y%m%d_%H%M%S)}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
NUM_TRAIN_CHUNKS="${NUM_TRAIN_CHUNKS:-4}"
NUM_HELD_OUT_CHUNKS="${NUM_HELD_OUT_CHUNKS:-4}"

cd "$PROJECT_DIR"

# Prepare data: use README as train, HANDOFF as held-out
TRAIN_FILE="README.md"
HELDOUT_FILE="HANDOFF.md"

echo "=== Layer-wise calibration sweep ==="
echo "Run ID: $RUN_ID"
echo "Model: $MODEL_PATH"
echo "Train: $TRAIN_FILE"
echo "Held-out: $HELDOUT_FILE"
echo ""

mkdir -p "$OUTPUT_DIR"

# Sweep configurations
# Strategy: later layers are closer to logits, calibrate those first
# Reduced configs to fit 24GB: smaller batch (max_tokens), fewer codes
CONFIGS=(
    "last-quarter:48:16:100:0.02"      # Only last 25% layers, minimal codes
    "last-quarter:64:24:100:0.02"      # Last 25%, slightly more codes
    "last-half:48:16:120:0.02"         # Last 50% layers, minimal
    "last-half:64:24:120:0.01"         # Last 50%, more codes, lower lr
    "last-half:48:32:150:0.01"         # Last 50%, moderate codes
)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r layers window codes steps lr <<< "$config"

    output_file="$OUTPUT_DIR/${RUN_ID}_layers-${layers}_w${window}_c${codes}_s${steps}.pt"
    log_file="$OUTPUT_DIR/${RUN_ID}_layers-${layers}_w${window}_c${codes}_s${steps}.log"

    echo ">>> Running: layers=$layers window=$window codes=$codes steps=$steps lr=$lr"

    CUDA_VISIBLE_DEVICES=0 \
    HF_ENDPOINT="$HF_ENDPOINT" \
    PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}" \
    python3 benchmarks/calibrate_hf_layerwise.py \
        --model "$MODEL_PATH" \
        --train-file "$TRAIN_FILE" \
        --held-out-file "$HELDOUT_FILE" \
        --output "$output_file" \
        --calibrate-layers "$layers" \
        --window-size "$window" \
        --num-codes "$codes" \
        --steps "$steps" \
        --lr "$lr" \
        --max-tokens 1024 \
        --num-train-chunks "$NUM_TRAIN_CHUNKS" \
        --num-held-out-chunks "$NUM_HELD_OUT_CHUNKS" \
        --kv-head-policy repeat \
        --run-id "$RUN_ID" \
        --quality-gate 0.99 \
        2>&1 | tee "$log_file"

    echo ""
done

echo "=== Sweep complete ==="
echo "Results in: $OUTPUT_DIR"
echo ""
echo "Summary:"
grep -h "held_out_gate_passed" "$OUTPUT_DIR"/${RUN_ID}_*.log | tail -6
