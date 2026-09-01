# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R001 | M0 | short retrieval sanity | QCC vs Full-KV | synthetic 64-token | accuracy, cosine | MUST | DONE | Existing synthetic smoke reached 1.000 accuracy. |
| R002 | M1 | CUDA kernel equivalence | dense/sparse Triton | deterministic tests | max state/output error | MUST | DONE | Colab T4: 24 passed. |
| R003 | M1 | matched latency control | QCC vs Full-KV | random 16K | TTFT, TPOT, cosine | MUST | DONE | T4: 1.36x TTFT, 2.70x TPOT, random cosine >0.998. |
| R004 | M2 | long stream | QCC | synthetic 128K | prefill, TPOT, state | MUST | DONE | T4: 3.831 s, 71.8 ms, 0.125781%. |
| R005 | M3 | long stream | QCC | synthetic 1M | prefill, TPOT, state | MUST | DONE | T4: 14.312 s, 72.4 ms, 0.0161%. |
| R006 | M3 | long stream | QCC | synthetic 4M | prefill, TPOT, state | MUST | DONE | T4: 54.564 s, 71.7 ms, 0.004025%. |
| R007 | M2 | trained long retrieval | QCC checkpoint | RULER 128K | accuracy | MUST | TODO | Requires checkpoint and RULER JSONL. |
| R008 | M3 | million-token retrieval | QCC checkpoint | retrieval 1M | accuracy, speed | MUST | TODO | Requires checkpoint and dataset. |
| R009 | M3 | novelty isolation | codebook/scale ablations | 128K/1M | accuracy-speed Pareto | MUST | TODO | Run only after R007/R008 data path works. |
