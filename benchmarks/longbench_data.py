#!/usr/bin/env python3
"""Official LongBench data: download-on-demand cache and record loading.

LongBench (Bai et al., 2023, https://github.com/THUDM/LongBench) is the
real-document long-context suite used here as the non-synthetic complement to
RULER.  This module owns everything about *data*:

* it downloads the official records into a local cache on demand
  (``$QCC_LONGBENCH_DIR``, default ``~/.cache/qcc/longbench``; ``--cache-dir``
  overrides both),
* it downloads the official prompt templates (``dataset2prompt.json``), the
  official per-task generation lengths (``dataset2maxlen.json``) and the
  task -> metric table (``dataset2metric.json``),
* it loads the records of a requested task list and builds the prompt exactly
  the way the official harness does
  (``template.format(context=..., input=...)``).

Upstream layout (checked 2026-09-21)
------------------------------------
``huggingface.co/datasets/THUDM/LongBench`` holds ``LongBench.py``, ``README.md``
and ``data.zip``; the zip contains ``data/<task>.jsonl`` plus the LongBench-E
variants ``data/<task>_e.jsonl``.  The three config JSONs are *not* in the HF
repo -- they live in the GitHub repository under ``LongBench/config/`` -- and
``dataset2metric.json`` does not exist upstream at all (the official ``eval.py``
hard-codes the table).  The loader therefore tries HF first (so it keeps working
if the repo is reorganised), then the GitHub raw files, and finally writes a
``dataset2metric.json`` synthesised from the official ``eval.py`` table shipped
in :mod:`longbench_metrics`, so the cache is self-describing either way.

Network note
------------
This machine's ``no_proxy`` contains ``[::1]``, which makes ``httpx`` (and hence
``huggingface_hub``) raise ``InvalidURL: Invalid port: ':1]'``.  The downloader
sanitises both spellings of the variable before touching the Hub.

CLI::

    python benchmarks/longbench_data.py --tasks narrativeqa qasper --report
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

ENV_CACHE_DIR = "QCC_LONGBENCH_DIR"
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "qcc" / "longbench")
HF_REPO_ID = "THUDM/LongBench"
CONFIG_RAW_BASE = "https://raw.githubusercontent.com/THUDM/LongBench/main/LongBench/config"

ARCHIVE_NAME = "data.zip"
PROMPT_FILE = "dataset2prompt.json"
MAXLEN_FILE = "dataset2maxlen.json"
METRIC_FILE = "dataset2metric.json"

#: The 21 official English/Chinese LongBench tasks (no LongBench-E variants).
ALL_TASKS = (
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa",
    "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news",
    "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count",
    "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p",
)

#: Sensible default: the English, real-document, non-code tasks whose official
#: metrics are implemented dependency-free in :mod:`longbench_metrics`.
#: ``vcsum`` is listed because the prompt asks for it; the runner skips it with
#: an explicit reason when ``jieba`` is unavailable (see benchmarks/README-longbench.md).
DEFAULT_TASKS = (
    "narrativeqa", "qasper", "hotpotqa", "2wikimqa", "gov_report", "multi_news",
    "triviaqa", "samsum", "vcsum", "passage_retrieval_en",
)

#: Every official record carries these fields.
REQUIRED_FIELDS = ("input", "context", "answers", "length", "dataset", "_id",
                   "all_classes")

MANIFEST_NAME = "manifest.json"


def sanitize_proxy_env() -> None:
    """Remove the ``[::1]`` entry that breaks ``httpx`` URL parsing.

    ``no_proxy=...,[::1]`` is not a valid host:port list for httpx and every
    request raises ``InvalidURL: Invalid port: ':1]'``.  Only the two no-proxy
    spellings are rewritten; real proxy settings are left untouched.
    """
    import os

    for name in ("NO_PROXY", "no_proxy"):
        value = os.environ.get(name)
        if value and "[::1]" in value:
            os.environ[name] = "localhost,127.0.0.1"


sanitize_proxy_env()  # import-time fix, as recommended by the harness notes


def resolve_cache_dir(path: str | Path | None = None) -> Path:
    """``argument > QCC_LONGBENCH_DIR > ~/.cache/qcc/longbench``."""
    import os

    if path is not None:
        return Path(path).expanduser()
    env = os.environ.get(ENV_CACHE_DIR)
    return Path(env).expanduser() if env else Path(DEFAULT_CACHE_DIR)


def task_path(task: str, cache_dir: str | Path | None = None) -> Path:
    return resolve_cache_dir(cache_dir) / "data" / f"{task}.jsonl"


def downloaded_tasks(cache_dir: str | Path | None = None) -> list[str]:
    data_dir = resolve_cache_dir(cache_dir) / "data"
    if not data_dir.is_dir():
        return []
    return sorted(p.stem for p in data_dir.glob("*.jsonl") if not p.stem.endswith("_e"))


# --------------------------------------------------------------------------- #
# download helpers
# --------------------------------------------------------------------------- #
def _fetch_url(url: str, timeout: float = 60.0) -> bytes:
    """Fetch ``url`` with the ambient proxy settings, retrying without them."""
    errors = []
    for handlers in (None, {}):  # first: environment proxies, then: direct
        opener = (urllib.request.build_opener(urllib.request.ProxyHandler(handlers))
                  if handlers is not None else urllib.request.build_opener())
        try:
            with opener.open(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError(f"could not fetch {url} ({'; '.join(errors)})")


def _hub_download(filename: str, cache_dir: Path, repo_id: str) -> Path | None:
    """One file from the Hub into ``cache_dir``; ``None`` when it is absent."""
    sanitize_proxy_env()
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep of the runner
        return None
    try:
        local = hf_hub_download(repo_id, filename, repo_type="dataset",
                                local_dir=str(cache_dir))
    except (EntryNotFoundError, HfHubHTTPError, OSError, ValueError):
        return None
    return Path(local)


def _hub_snapshot(cache_dir: Path, repo_id: str, patterns: list[str]) -> bool:
    """Best-effort ``snapshot_download`` of the expected per-file layout."""
    sanitize_proxy_env()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:  # pragma: no cover
        return False
    try:
        snapshot_download(repo_id, repo_type="dataset", local_dir=str(cache_dir),
                          allow_patterns=patterns)
        return True
    except Exception:  # pragma: no cover - the archive fallback covers every failure
        return False


def _install_config(name: str, cache_dir: Path, repo_id: str,
                    verbose: bool) -> tuple[Path, str]:
    """Materialise one config JSON; returns ``(path, source)``."""
    target = cache_dir / name
    if target.exists() and target.stat().st_size > 0:
        return target, "cache"
    fetched = _hub_download(name, cache_dir, repo_id)
    if fetched is not None and fetched.exists() and fetched.stat().st_size > 0:
        if fetched != target:
            target.write_bytes(fetched.read_bytes())
        return target, "huggingface"
    try:
        payload = _fetch_url(f"{CONFIG_RAW_BASE}/{name}")
        json.loads(payload)  # validate before writing
        target.write_bytes(payload)
        return target, "github"
    except Exception as exc:
        if name == METRIC_FILE:
            # dataset2metric.json does not exist upstream: synthesise it from the
            # official eval.py table so the cache is self-describing.
            from longbench_metrics import OFFICIAL_DATASET2METRIC

            target.write_text(json.dumps(OFFICIAL_DATASET2METRIC, indent=1) + "\n")
            if verbose:
                print(f"[longbench] {name}: not upstream ({type(exc).__name__}); "
                      f"wrote the official eval.py table instead", flush=True)
            return target, "synthesised-from-eval.py"
        raise


def _extract_tasks(archive: Path, cache_dir: Path, tasks: list[str],
                   verbose: bool) -> None:
    """Extract only ``data/<task>.jsonl`` for the requested tasks."""
    data_dir = cache_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"data/{task}.jsonl": task for task in tasks}
    with zipfile.ZipFile(archive) as zf:
        available = set(zf.namelist())
        missing = [member for member in wanted if member not in available]
        if missing:
            raise KeyError(f"{archive} does not contain {sorted(missing)}")
        for member, task in wanted.items():
            target = data_dir / f"{task}.jsonl"
            if target.exists() and target.stat().st_size > 0:
                continue
            temp = data_dir / f"{task}.jsonl.part"
            with zf.open(member) as source, open(temp, "wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 20)
            temp.replace(target)
            if verbose:
                print(f"[longbench] extracted data/{task}.jsonl "
                      f"({target.stat().st_size / 2**20:.1f} MiB)", flush=True)


def _validate_task_file(path: Path, task: str) -> int:
    """Read every line, check the official fields, return the record count."""
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not JSON: {exc}") from exc
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                raise ValueError(f"{path}:{line_no} misses {missing}")
            if not isinstance(record["answers"], list):
                raise ValueError(f"{path}:{line_no} 'answers' is not a list")
            count += 1
    if count == 0:
        raise ValueError(f"{path} contains no records")
    return count


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def ensure_dataset(cache_dir: str | Path | None = None,
                   tasks: list[str] | tuple[str, ...] | None = None,
                   *,
                   repo_id: str = HF_REPO_ID,
                   refresh: bool = False,
                   keep_archive: bool = True,
                   verbose: bool = True) -> dict:
    """Download whatever is missing and return a manifest describing the cache.

    The manifest is also written to ``<cache_dir>/manifest.json`` and contains
    the per-task record counts, the config paths and the download sources, so a
    later run (and the JSON report of a benchmark run) can quote exactly which
    data was used.
    """
    cache = resolve_cache_dir(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    # ``tasks=None`` means "the whole suite"; ``tasks=[]`` means "configs only".
    wanted = list(ALL_TASKS) if tasks is None else list(tasks)
    unknown = [task for task in wanted if task not in ALL_TASKS]
    if unknown:
        raise ValueError(f"unknown LongBench task(s) {unknown}; "
                         f"known tasks: {list(ALL_TASKS)}")

    # 1) the official per-file layout, in case the Hub repo grows one
    _hub_snapshot(cache, repo_id, ["data/*.jsonl", "*.json"])

    # 2) configs
    prompts, prompt_source = _install_config(PROMPT_FILE, cache, repo_id, verbose)
    maxlens, maxlen_source = _install_config(MAXLEN_FILE, cache, repo_id, verbose)
    metrics, metric_source = _install_config(METRIC_FILE, cache, repo_id, verbose)

    # 3) records: extract only what is missing from the official archive
    missing = [task for task in wanted
               if refresh or not task_path(task, cache).exists()
               or task_path(task, cache).stat().st_size == 0]
    archive = cache / "_download" / ARCHIVE_NAME
    archive_used = False
    if missing:
        if refresh and archive.exists():
            archive.unlink()
        if not archive.exists():
            if verbose:
                print(f"[longbench] downloading {repo_id}/{ARCHIVE_NAME} ...", flush=True)
            started = time.time()
            fetched = _hub_download(ARCHIVE_NAME, cache / "_download", repo_id)
            if fetched is None:
                raise RuntimeError(
                    f"could not download {ARCHIVE_NAME} from {repo_id}; check the network "
                    f"(no_proxy must not contain '[::1]')")
            if verbose:
                print(f"[longbench] archive {archive.stat().st_size / 2**20:.1f} MiB in "
                      f"{time.time() - started:.1f}s", flush=True)
        _extract_tasks(archive, cache, missing, verbose)
        archive_used = True
        if not keep_archive and archive.exists():
            archive.unlink()

    # 4) validate and count
    tasks_report: dict[str, dict] = {}
    for task in wanted:
        path = task_path(task, cache)
        if not path.exists():
            raise FileNotFoundError(f"missing LongBench task file {path}")
        tasks_report[task] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "records": _validate_task_file(path, task),
        }

    manifest = {
        "schema": "qcc-longbench-cache-v1",
        "repo_id": repo_id,
        "cache_dir": str(cache),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tasks": tasks_report,
        "configs": {
            PROMPT_FILE: {"path": str(prompts), "source": prompt_source},
            MAXLEN_FILE: {"path": str(maxlens), "source": maxlen_source},
            METRIC_FILE: {"path": str(metrics), "source": metric_source},
        },
        "archive": {"path": str(archive), "used": archive_used,
                    "kept": archive.exists()},
        "total_records": sum(entry["records"] for entry in tasks_report.values()),
    }
    (cache / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def load_prompt_templates(cache_dir: str | Path | None = None,
                          auto_download: bool = True) -> dict:
    cache = resolve_cache_dir(cache_dir)
    path = cache / PROMPT_FILE
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(f"{path} is missing; run ensure_dataset() first")
        ensure_dataset(cache, tasks=(), verbose=False)
    return json.loads(path.read_text(encoding="utf-8"))


def load_max_lengths(cache_dir: str | Path | None = None,
                     auto_download: bool = True) -> dict:
    cache = resolve_cache_dir(cache_dir)
    path = cache / MAXLEN_FILE
    if not path.exists():
        if not auto_download:
            raise FileNotFoundError(f"{path} is missing; run ensure_dataset() first")
        ensure_dataset(cache, tasks=(), verbose=False)
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(tasks: list[str] | tuple[str, ...] | None = None,
                 cache_dir: str | Path | None = None,
                 limit: int | None = None,
                 auto_download: bool = True,
                 verbose: bool = False) -> list[dict]:
    """Records of ``tasks`` in the official file order, ``limit`` per task.

    Every record is returned verbatim; it carries ``dataset`` (the task name),
    ``_id``, ``length`` (official GPT-3.5-token length), ``context``, ``input``,
    ``answers`` and ``all_classes``.
    """
    wanted = list(tasks) if tasks else list(DEFAULT_TASKS)
    cache = resolve_cache_dir(cache_dir)
    if auto_download:
        ensure_dataset(cache, tasks=wanted, verbose=verbose)
    records: list[dict] = []
    for task in wanted:
        path = task_path(task, cache)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; call ensure_dataset(tasks=[...]) or drop "
                f"--auto-download")
        taken = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if limit is not None and taken >= limit:
                    break
                record = json.loads(line)
                record.setdefault("dataset", task)
                records.append(record)
                taken += 1
        if verbose:
            print(f"[longbench] {task}: {taken} record(s)", flush=True)
    return records


def build_prompt(record: dict, templates: dict) -> str:
    """Official prompt construction: ``template.format(context=..., input=...)``.

    ``str.format(**record)`` is what the official ``pred.py`` does, and every
    official template only references ``{context}`` and ``{input}``; formatting
    with the whole record keeps that behaviour and stays compatible if a future
    template adds a field.
    """
    task = record.get("dataset")
    if task not in templates:
        raise KeyError(f"no prompt template for task {task!r}")
    return templates[task].format(**record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="*", default=list(DEFAULT_TASKS))
    parser.add_argument("--cache-dir", default=None,
                        help=f"default: ${ENV_CACHE_DIR} or {DEFAULT_CACHE_DIR}")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true",
                        help="re-download the archive and re-extract every task")
    parser.add_argument("--prune-archive", action="store_true",
                        help="delete data.zip after extraction")
    parser.add_argument("--report", action="store_true",
                        help="print one sample prompt per task")
    args = parser.parse_args(argv)

    manifest = ensure_dataset(args.cache_dir, args.tasks, refresh=args.refresh,
                              keep_archive=not args.prune_archive)
    print(json.dumps({key: manifest[key] for key in
                      ("cache_dir", "total_records", "configs", "archive")}, indent=1))
    print(f"{'task':<22}{'records':>8}{'MiB':>9}")
    for task, entry in manifest["tasks"].items():
        print(f"{task:<22}{entry['records']:>8}{entry['bytes'] / 2**20:>9.1f}")

    if args.report:
        records = load_records(args.tasks, args.cache_dir, limit=args.limit,
                               auto_download=False)
        templates = load_prompt_templates(args.cache_dir)
        for record in records[:len(args.tasks)]:
            prompt = build_prompt(record, templates)
            print(f"\n=== {record['dataset']} ({record['_id']}) "
                  f"prompt_chars={len(prompt)} answers={record['answers'][:2]} ===")
            print(prompt[:600] + (" ..." if len(prompt) > 600 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
