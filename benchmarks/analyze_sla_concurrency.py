"""Max concurrent requests under a per-request TPOT SLA, Full-KV vs bounded."""
import json, sys

def load(path):
    try:
        return json.load(open(path))["results"]
    except Exception:
        return []

def best_batch(rows, policy, sla_ms):
    """`policy` may be None to accept rows that carry no policy field (the
    sequential serving harness omits it)."""
    ok = []
    for e in rows:
        if e.get("status") != "ok" or e.get("tpot_ms") is None:
            continue
        if policy is not None and e.get("policy") not in (None, policy):
            continue
        if e["tpot_ms"] <= sla_ms:
            ok.append(e["batch"])
    return max(ok, default=0)

cases = [
    ("32K, B=1024 (speed)", "serving_32k_v2.json", "serving_seq_32k.json"),
    ("32K, B=4096 (quality)", "serving_32k_v2.json", "serving_seq_32k_q4096.json"),
    ("128K, B=1024", "serving_128k_matched.json", "serving_seq_128k.json"),
    ("128K, B=4096 (quality)", "serving_128k_matched.json", "serving_seq_128k_q4096.json"),
]
print(f"{'case':<24}{'SLA':>6}{'full-kv':>9}{'bounded':>9}{'ratio':>8}")
for label, full_path, bounded_path in cases:
    full_rows, bounded_rows = load(full_path), load(bounded_path)
    if not full_rows or not bounded_rows:
        print(f"{label:<24}  (missing data: {full_path} / {bounded_path})")
        continue
    for sla in (25, 50, 100):
        f = best_batch(full_rows, "full", sla)
        b = best_batch(bounded_rows, None, sla)
        ratio = f"{b / f:.1f}x" if f else ("inf" if b else "-")
        print(f"{label:<24}{sla:>6}{f:>9}{b:>9}{ratio:>8}")
