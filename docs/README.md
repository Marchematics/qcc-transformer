# Documentation

| page | audience | contents |
|---|---|---|
| [`REPORT.md`](REPORT.md) | reviewers | the full measurement record: every claim, its table, and the conditions each number was taken under, including what is explicitly *not* established |
| [`CLAIMS.md`](CLAIMS.md) | reviewers, maintainers | claim -> shipped code path -> artifact -> command, plus the open questions and the experiment that closes each |
| [`REPRODUCING.md`](REPRODUCING.md) | anyone re-running | setup, the command behind each artifact, and the hardware limits |
| [`MECHANISM.md`](MECHANISM.md) | reviewers, theorists | why a fixed support can be enough, the measured NLL-vs-budget curves, and three falsifiable predictions with the experiments that would test them |
| [`NOVELTY.md`](NOVELTY.md) | reviewers | the boundary against prior bounded-memory attention work |
| [`RELEASE.md`](RELEASE.md) | users | what is in the snapshot bundle and how to verify it |
| [`HANDOFF.zh.md`](HANDOFF.zh.md) | the next engineer | working log in Chinese: current audit, bugs found, gaps, push procedure |
| [`legacy/`](legacy/) | historians | the pre-reorganisation README, workspace notes and experiment logs |

The measured evidence lives in [`../artifacts/`](../artifacts): one JSON per run,
each carrying per-record predictions, scores, retained slot counts, timings and
peak memory, so any table can be recomputed without a GPU.
