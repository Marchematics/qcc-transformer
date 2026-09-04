#!/usr/bin/env bash
set -euo pipefail

# Prepare a DSW GPU workspace without downloading model weights implicitly.
ROOT="${1:-$(pwd)}"
cd "$ROOT"

echo '== GPU =='
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

echo '== Python =='
python --version
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
PY

echo '== Install editable package =='
python -m pip install -e '.[hf]' --no-input

mkdir -p artifacts/remote_gpu/dsw_runs
echo '== Candidate checkpoints =='
find /mnt/workspace /mnt/data /root/.cache/huggingface -maxdepth 5 \
  -type f \( -name config.json -o -name '*.safetensors' -o -name '*.bin' \) \
  2>/dev/null | sed -n '1,80p' || true

echo '== Existing processes =='
ps -eo pid,stat,etime,cmd | grep -E 'python|vllm|qcc' | grep -v grep | head -80 || true

echo "Prepared $(date -Is) in $ROOT"
