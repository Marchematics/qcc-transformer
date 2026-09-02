# Remote experiment workspace

The shared GPU host keeps the active QCC workspace at:

```text
/home/frankwang122222/zjh/工作目录/工作文件
```

This account cannot create `/zjh` itself, so the home-scoped path is the
canonical location for this project. Source files are synchronized there from
the repository; checkpoints, generated datasets, and long logs stay outside
the Git working tree unless a small summary is intentionally committed.

Cleanup policy:

- Python caches and test caches may be removed after runs.
- Archives are never deleted solely by filename. Before removal, verify that
  they belong to this project, are not referenced by an active run, and are
  older than the retention window.
- The existing `HOTC2026/临时归档` ZIP files are retained because they are
  user data created on the current date, not QCC experiment artifacts.
