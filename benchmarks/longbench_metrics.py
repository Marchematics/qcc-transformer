#!/usr/bin/env python3
"""Official LongBench metrics, reimplemented dependency-free.

The reference implementation is ``THUDM/LongBench``'s ``LongBench/metrics.py``
and ``LongBench/eval.py`` (fetched 2026-09-21).  Everything here mirrors those
files; nothing imports ``rouge``, ``jieba``, ``fuzzywuzzy`` or ``nltk``.

Implemented (all faithful to the official code):

``qa_f1_score``
    SQuAD-style token F1 after the official ``normalize_answer`` (lowercase,
    drop ``string.punctuation``, drop the articles ``a|an|the``, collapse
    whitespace) and the official ``f1_score`` (multiset intersection over
    token lists).
``rouge_score`` / ``rouge_l``
    ROUGE-L *summary level*, i.e. the ``rouge`` PyPI package
    (pltrdy/rouge) as called by ``LongBench/metrics.py``:
    ``Rouge().get_scores([pred], [gt], avg=True)["rouge-l"]["f"]``.  That is
    the sentence-level union-LCS variant: both texts are split on ``"."``,
    each reference sentence is aligned against *all* hypothesis sentences,
    the per-sentence LCS token sets are unioned, ``m``/``n`` are the counts of
    *distinct* reference/hypothesis words, and
    ``f = 2*P*R/(P+R+1e-8)``.  The official wrapper swallows every exception
    and returns ``0.0``; so does this one (empty prediction -> ``0.0``).
``classification_score``
    Exact-match-over-classes accuracy for ``trec``/``lsht``, including the
    official loop that removes from the match list *while iterating it*.
``retrieval_score`` / ``retrieval_zh_score``
    Fraction of numbers in the prediction equal to the ``Paragraph N`` /
    ``段落N`` id of the ground truth.  Note the official definition: a
    prediction with the right id plus unrelated numbers scores < 1.
``count_score``
    Fraction of numbers in the prediction equal to the ground-truth count
    (``passage_count``).

Not implemented (they need a third-party dependency that is not installed and
whose absence the official harness would also notice): ``rouge_zh_score`` and
``qa_f1_zh_score`` (``jieba``), ``code_sim_score`` (``fuzzywuzzy``).  Asking for
one of those tasks raises :class:`MetricUnavailableError` with the missing
package named; :func:`metric_available` lets a runner skip such tasks cleanly.

CLI::

    python benchmarks/longbench_metrics.py    # task -> metric -> available?
"""
from __future__ import annotations

import itertools
import json
import re
import string
from pathlib import Path

__all__ = [
    "OFFICIAL_DATASET2METRIC",
    "MetricUnavailableError",
    "TASK_FIRST_LINE_PREDICTION",
    "classification_score",
    "count_score",
    "f1_score",
    "load_dataset2metric",
    "metric_available",
    "metric_for_task",
    "normalize_answer",
    "normalize_zh_answer",
    "qa_f1_score",
    "retrieval_score",
    "retrieval_zh_score",
    "rouge_l",
    "rouge_l_summary_level",
    "rouge_score",
    "score_record",
    "unavailable_reason",
]

#: ``eval.py``'s hard-coded ``dataset2metric`` table, as metric *names*.
OFFICIAL_DATASET2METRIC: dict[str, str] = {
    "narrativeqa": "qa_f1_score",
    "qasper": "qa_f1_score",
    "multifieldqa_en": "qa_f1_score",
    "multifieldqa_zh": "qa_f1_zh_score",
    "hotpotqa": "qa_f1_score",
    "2wikimqa": "qa_f1_score",
    "musique": "qa_f1_score",
    "dureader": "rouge_zh_score",
    "gov_report": "rouge_score",
    "qmsum": "rouge_score",
    "multi_news": "rouge_score",
    "vcsum": "rouge_zh_score",
    "trec": "classification_score",
    "triviaqa": "qa_f1_score",
    "samsum": "rouge_score",
    "lsht": "classification_score",
    "passage_retrieval_en": "retrieval_score",
    "passage_count": "count_score",
    "passage_retrieval_zh": "retrieval_zh_score",
    "lcc": "code_sim_score",
    "repobench-p": "code_sim_score",
}

#: ``eval.py`` scores only the first line of these tasks' predictions.
TASK_FIRST_LINE_PREDICTION = frozenset({"trec", "triviaqa", "samsum", "lsht"})

#: Third-party imports the official metric needs, when any.
METRIC_REQUIREMENTS = {
    "rouge_zh_score": "jieba",
    "qa_f1_zh_score": "jieba",
    "code_sim_score": "fuzzywuzzy",
}


class MetricUnavailableError(RuntimeError):
    """Raised when an official metric needs a package that is not installed."""


# --------------------------------------------------------------------------- #
# qa_f1_score
# --------------------------------------------------------------------------- #
def normalize_answer(text: str) -> str:
    """Official ``metrics.normalize_answer`` (SQuAD normalisation)."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def normalize_zh_answer(text: str) -> str:
    """Official ``metrics.normalize_zh_answer`` (punctuation + whitespace)."""
    cn_punctuation = ("！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」"
                      "『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏.")
    all_punctuation = set(string.punctuation + cn_punctuation)
    stripped = "".join(char for char in text.lower() if char not in all_punctuation)
    return "".join(stripped.split())


def f1_score(prediction_tokens, ground_truth_tokens) -> float:
    """Official ``metrics.f1_score``: multiset F1 over token sequences.

    Note that the official code passes sequences (lists), not sets:
    ``Counter(prediction) & Counter(ground_truth)`` counts repeated tokens
    ``min(count_pred, count_gt)`` times.
    """
    from collections import Counter

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.qa_f1_score``."""
    prediction_tokens = normalize_answer(str(prediction)).split()
    ground_truth_tokens = normalize_answer(str(ground_truth)).split()
    return float(f1_score(prediction_tokens, ground_truth_tokens))


# --------------------------------------------------------------------------- #
# rouge_score (ROUGE-L summary level, pltrdy/rouge semantics)
# --------------------------------------------------------------------------- #
def _sentences(text: str) -> list[str]:
    """``Rouge._get_avg_scores``: split on ``.``, drop empty, collapse spaces."""
    return [" ".join(part.split()) for part in str(text).split(".") if len(part) > 0]


def _words(sentences) -> list[str]:
    """``rouge_score._split_into_words``: split on a single space, flattened."""
    return list(itertools.chain(*[sentence.split(" ") for sentence in sentences]))


def _lcs_table(x, y):
    """``rouge_score._lcs``: full DP table of LCS lengths."""
    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, previous = table[i], table[i - 1]
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                row[j] = previous[j - 1] + 1
            else:
                row[j] = previous[j] if previous[j] >= row[j - 1] else row[j - 1]
    return table


def _recon_lcs(x, y) -> list:
    """``rouge_score._recon_lcs``: reconstruct one LCS (same tie-breaking)."""
    table = _lcs_table(x, y)
    result = []
    i, j = len(x), len(y)
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            result.append(x[i - 1])
            i -= 1
            j -= 1
        elif table[i - 1][j] > table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return result[::-1]


def rouge_l_summary_level(evaluated_sentences, reference_sentences) -> dict:
    """``rouge_score.rouge_l_summary_level(..., exclusive=True)``.

    ``evaluated_sentences``/``reference_sentences`` are lists of sentences (the
    caller sentence-splits on ``"."``).  Returns ``{"f", "p", "r"}``.
    """
    if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
        raise ValueError("Collections must contain at least 1 sentence.")

    # Number of *distinct* words (Ngrams with exclusive=True is a set).
    m = len(set(_words(reference_sentences)))
    n = len(set(_words(evaluated_sentences)))

    union: set[str] = set()
    union_lcs_sum = 0
    for reference_sentence in reference_sentences:
        reference_words = _words([reference_sentence])
        sentence_union: set[str] = set()
        for evaluated_sentence in evaluated_sentences:
            evaluated_words = _words([evaluated_sentence])
            sentence_union |= set(_recon_lcs(reference_words, evaluated_words))
        union_lcs_sum += len(union | sentence_union) - len(union)
        union |= sentence_union

    recall = union_lcs_sum / m
    precision = union_lcs_sum / n
    return {"f": 2.0 * ((precision * recall) / (precision + recall + 1e-8)),
            "p": precision, "r": recall}


def rouge_l(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.rouge_score``: F of the ``rouge`` package's ROUGE-L.

    The official wrapper is ``try: ... except: return 0.0`` around
    ``Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"]``,
    so *any* failure -- an empty prediction or an empty reference above all --
    scores ``0.0`` rather than raising.
    """
    try:
        hypothesis = _sentences(prediction)
        reference = _sentences(ground_truth)
        return float(rouge_l_summary_level(hypothesis, reference)["f"])
    except Exception:
        return 0.0


#: ``eval.py`` calls this metric ``rouge_score``.
rouge_score = rouge_l


# --------------------------------------------------------------------------- #
# classification / retrieval / count
# --------------------------------------------------------------------------- #
def classification_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Official ``metrics.classification_score`` for ``trec`` and ``lsht``.

    Faithful in one ugly detail: the official loop removes from
    ``em_match_list`` while iterating over it, so a prediction containing
    several classes can skip a removal.  Reproducing it keeps the scores
    comparable with published LongBench tables.
    """
    all_classes = kwargs.get("all_classes") or []
    em_match_list = []
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return float(1.0 / len(em_match_list))
    return 0.0


def _numbered_match(prediction: str, ground_truth: str, pattern: str) -> float:
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(0.0 if len(numbers) == 0 else right_num / len(numbers))


def retrieval_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.retrieval_score`` (``passage_retrieval_en``)."""
    return _numbered_match(prediction, ground_truth, r"Paragraph (\d+)")


def retrieval_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.retrieval_zh_score`` (``passage_retrieval_zh``)."""
    return _numbered_match(prediction, ground_truth, r"段落(\d+)")


def count_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.count_score`` (``passage_count``)."""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(0.0 if len(numbers) == 0 else right_num / len(numbers))


# --------------------------------------------------------------------------- #
# metrics that need a package the official harness also requires
# --------------------------------------------------------------------------- #
def rouge_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.rouge_zh_score``: jieba segmentation + ROUGE-L."""
    jieba = _require("jieba", "rouge_zh_score")
    hypothesis = " ".join(jieba.cut(str(prediction), cut_all=False))
    reference = " ".join(jieba.cut(str(ground_truth), cut_all=False))
    return rouge_l(hypothesis, reference)


def qa_f1_zh_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.qa_f1_zh_score``: jieba segmentation + token F1."""
    jieba = _require("jieba", "qa_f1_zh_score")
    prediction_tokens = [normalize_zh_answer(token)
                         for token in jieba.cut(str(prediction), cut_all=False)]
    ground_truth_tokens = [normalize_zh_answer(token)
                           for token in jieba.cut(str(ground_truth), cut_all=False)]
    return float(f1_score([token for token in prediction_tokens if token],
                          [token for token in ground_truth_tokens if token]))


def code_sim_score(prediction: str, ground_truth: str, **_: object) -> float:
    """Official ``metrics.code_sim_score``: first code line vs reference, fuzz."""
    _require("fuzzywuzzy", "code_sim_score")
    from fuzzywuzzy import fuzz

    first_line = ""
    for line in str(prediction).lstrip("\n").split("\n"):
        if ("`" not in line) and ("#" not in line) and ("//" not in line):
            first_line = line
            break
    return float(fuzz.ratio(first_line, ground_truth) / 100)


def _require(module: str, metric: str):
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MetricUnavailableError(
            f"metric {metric!r} needs the {module!r} package ({exc}); "
            f"install it or drop that task") from exc


METRIC_REGISTRY = {
    "qa_f1_score": qa_f1_score,
    "rouge_score": rouge_score,
    "rouge_l": rouge_l,
    "classification_score": classification_score,
    "retrieval_score": retrieval_score,
    "retrieval_zh_score": retrieval_zh_score,
    "count_score": count_score,
    "rouge_zh_score": rouge_zh_score,
    "qa_f1_zh_score": qa_f1_zh_score,
    "code_sim_score": code_sim_score,
}


# --------------------------------------------------------------------------- #
# task -> metric plumbing
# --------------------------------------------------------------------------- #
def _default_dataset2metric_path() -> Path | None:
    try:
        from longbench_data import METRIC_FILE, resolve_cache_dir
    except ImportError:  # pragma: no cover - only when imported out of tree
        return None
    path = resolve_cache_dir(None) / METRIC_FILE
    return path if path.exists() else None


def load_dataset2metric(source: str | Path | dict | None = None) -> dict[str, str]:
    """Task -> metric-name table, from ``dataset2metric.json`` when available.

    ``source`` may be the JSON path, the cache directory holding it, or an
    already-loaded mapping.  Missing tasks fall back to the official ``eval.py``
    table (upstream has no ``dataset2metric.json``, so the file shipped in this
    cache is synthesised from that table by :mod:`longbench_data`).
    """
    table = dict(OFFICIAL_DATASET2METRIC)
    payload = source
    if payload is None:
        path = _default_dataset2metric_path()
        payload = json.loads(path.read_text(encoding="utf-8")) if path else None
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "dataset2metric.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if not payload:
        return table
    for task, metric in payload.items():
        if isinstance(metric, str) and metric not in METRIC_REGISTRY:
            raise ValueError(f"dataset2metric.json maps {task!r} to unknown metric "
                             f"{metric!r}; known: {sorted(METRIC_REGISTRY)}")
        table[task] = metric
    return table


def metric_for_task(task: str, mapping: dict | None = None):
    """The callable that scores ``task`` (raises for unsupported tasks)."""
    table = mapping if mapping is not None else load_dataset2metric()
    if task not in table:
        raise KeyError(f"no LongBench metric for task {task!r}")
    metric = table[task]
    if callable(metric):
        return metric
    if metric not in METRIC_REGISTRY:
        raise MetricUnavailableError(f"unknown metric {metric!r} for task {task!r}")
    return METRIC_REGISTRY[metric]


def unavailable_reason(task: str, mapping: dict | None = None) -> str | None:
    """``None`` when ``task`` can be scored here, else why it cannot."""
    import importlib.util

    table = mapping if mapping is not None else load_dataset2metric()
    metric = table.get(task)
    if metric is None:
        return f"no metric registered for task {task!r}"
    if callable(metric):
        return None
    package = METRIC_REQUIREMENTS.get(metric)
    if package is None:
        return None if metric in METRIC_REGISTRY else f"unknown metric {metric!r}"
    if importlib.util.find_spec(package) is None:
        return f"{metric} needs the {package!r} package, which is not installed"
    return None


def metric_available(task: str, mapping: dict | None = None) -> bool:
    return unavailable_reason(task, mapping) is None


def score_record(task: str, prediction: str, record: dict,
                 mapping: dict | None = None) -> float:
    """Official per-record score in ``[0, 1]``: max over the reference answers.

    Mirrors ``eval.py``'s ``scorer``: for ``trec``/``triviaqa``/``samsum``/
    ``lsht`` only the first predicted line is used, and the score of a record is
    the best score over its reference answers (0.0 when there are none).
    """
    metric = metric_for_task(task, mapping)
    text = "" if prediction is None else str(prediction)
    if task in TASK_FIRST_LINE_PREDICTION:
        text = text.lstrip("\n").split("\n")[0]
    answers = record.get("answers") or []
    all_classes = record.get("all_classes")
    best = 0.0
    for ground_truth in answers:
        score = metric(text, ground_truth, all_classes=all_classes)
        if score is not None and float(score) > best:
            best = float(score)
    return best


def main() -> int:
    table = load_dataset2metric()
    print(f"{'task':<22}{'metric':<22}available")
    for task in sorted(table):
        reason = unavailable_reason(task, table)
        print(f"{task:<22}{str(table[task]):<22}{'yes' if reason is None else reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
