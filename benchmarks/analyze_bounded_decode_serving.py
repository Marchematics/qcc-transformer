"""Compare batched-prefill and sequential-prefill serving results.

Usage: python analyze_serving.py serving_32k_v2.json serving_seq_32k.json
"""
import json, sys

def load(p):
    try:
        return json.load(open(p))["results"]
    except Exception:
        return []

def show(title, rows):
    print(f"== {title} ==")
    print(f"{'policy':<10}{'batch':>6}{'status':>7}{'tok/s':>10}{'TPOT ms':>9}{'peak GiB':>9}{'recall':>9}")
    for e in rows:
        if e.get("status") == "ok":
            r = e.get("recall") or []
            print(f"{e.get('policy','bounded'):<10}{e['batch']:>6}{'ok':>7}"
                  f"{e['decode_tokens_per_s']:>10.2f}{e['tpot_ms']:>9.2f}"
                  f"{e['peak_cuda_gib']:>9.2f}{(sum(r)/len(r) if r else float('nan')):>9.3f}")
        else:
            print(f"{e.get('policy','bounded'):<10}{e['batch']:>6}{e['status']:>7}")
    ok = [e for e in rows if e.get("status") == "ok"]
    if ok:
        print(f"  max batch that fit: {max(e['batch'] for e in ok)}")

for path in sys.argv[1:]:
    rows = load(path)
    if rows:
        show(path, rows)
