import json
import sys
from pathlib import Path

sys.path.insert(0, "/content/qcc-transformer")
from benchmarks.evaluate_retrieval import evaluate, _load_checkpoint
from qcc_transformer import QCCForCausalLM
import torch

root = Path("/content/qcc-transformer")
checkpoint = root / "artifacts/colab_none_prefix.pt"
dataset = root / "artifacts/colab_none_prefix_long.jsonl"
model = QCCForCausalLM(vocab_size=32, d_model=32, num_layers=1, num_heads=4,
    max_position_embeddings=1_000_001, window_size=16, num_codes=32,
    position_encoding="none", archive_persistent_landmark=True,
    archive_prefix_landmark=True).cuda().eval()
model.load_state_dict(_load_checkpoint(checkpoint), strict=True)
correct, total = evaluate(model, dataset, chunk_size=256, device=torch.device("cuda"), max_examples=None)
payload = {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}
(root / "artifacts/colab_none_prefix_result.json").write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps(payload), flush=True)
