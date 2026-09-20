#!/usr/bin/env python3
"""CPU-only unit tests for the LongBench metric reimplementations.

Every expected value below is computed by hand from the official code
(``THUDM/LongBench``'s ``metrics.py``/``eval.py`` plus the ``rouge`` package it
calls); none of these tests touch the network, a GPU or a model.

Run::

    pytest -q tests/test_longbench_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import longbench_data  # noqa: E402  (import also sanitises the broken no_proxy)
import longbench_metrics as lm  # noqa: E402

APPROX = dict(rel=1e-6, abs=1e-9)


# --------------------------------------------------------------------------- #
# qa_f1_score
# --------------------------------------------------------------------------- #
def test_qa_f1_perfect_match_after_official_normalisation():
    # lowercase, punctuation and the articles a/an/the are removed by
    # normalize_answer, so these two are the same bag of tokens.
    assert lm.qa_f1_score("The cat sat on the mat.", "the cat sat on the mat") == 1.0


def test_qa_f1_empty_prediction_is_zero():
    assert lm.qa_f1_score("", "the cat sat on the mat") == 0.0
    assert lm.qa_f1_score("   ", "anything") == 0.0
    assert lm.qa_f1_score("!!!", "anything") == 0.0


def test_qa_f1_partial_overlap_hand_computed():
    # prediction tokens ["cat"], ground truth ["cat", "sat"]:
    #   precision 1/1, recall 1/2 -> f1 = 2*1*0.5/1.5 = 2/3
    assert lm.qa_f1_score("the cat", "the cat sat") == pytest.approx(2 / 3, **APPROX)


def test_qa_f1_is_order_insensitive_and_counts_repeats():
    assert lm.qa_f1_score("mat the on sat cat", "cat sat on the mat") == 1.0
    # ["cat","cat","cat"] vs ["cat","cat"]: common 2, p=2/3, r=1 -> 0.8
    assert lm.qa_f1_score("cat cat cat", "cat cat") == pytest.approx(0.8, **APPROX)


def test_normalize_answer_matches_official_rules():
    assert lm.normalize_answer("A  Dog, the Cat!") == "dog cat"
    assert lm.normalize_answer("An APPLE") == "apple"


# --------------------------------------------------------------------------- #
# rouge_score / rouge_l  (sentence-level union LCS, no lowercasing)
# --------------------------------------------------------------------------- #
def test_rouge_l_perfect_match_is_one_up_to_the_official_epsilon():
    # The official f is 2PR/(P+R+1e-8), so a perfect match is 0.999999995.
    value = lm.rouge_l("the cat sat on the mat.", "the cat sat on the mat.")
    assert value == pytest.approx(1.0, abs=1e-7)
    assert value < 1.0


def test_rouge_l_empty_prediction_and_empty_reference_are_zero():
    assert lm.rouge_l("", "the cat sat on the mat.") == 0.0
    assert lm.rouge_l("the cat sat on the mat.", "") == 0.0
    assert lm.rouge_l(".", "the cat.") == 0.0


def test_rouge_l_sentence_level_union_lcs_hand_computed():
    """The worked example from the official ``_union_lcs`` docstring.

    reference sentence: w1 w2 w3 w4 w5
    hypothesis sentences: "w1 w2 w6 w7 w8" and "w1 w3 w8 w9 w5"
    LCSes: "w1 w2" and "w1 w3 w5" -> union {w1,w2,w3,w5} = 4
    m = 5 distinct reference words, n = 8 distinct hypothesis words
    P = 4/8, R = 4/5 -> f = 0.8/1.3 = 0.615384...
    """
    value = lm.rouge_l("w1 w2 w6 w7 w8. w1 w3 w8 w9 w5", "w1 w2 w3 w4 w5")
    assert value == pytest.approx(4 / 5 / (4 / 8 + 4 / 5), **APPROX)  # 2PR/(P+R)
    assert value == pytest.approx(0.615384615, abs=1e-6)


def test_rouge_l_short_hypothesis_recall_limited():
    # hypothesis "a b" covers only the first of two reference sentences:
    # union LCS = 2, m = 4, n = 2 -> P = 1, R = 0.5 -> f = 2/3
    assert lm.rouge_l("a b", "a b. c d") == pytest.approx(2 / 3, **APPROX)


def test_rouge_l_is_case_sensitive_unlike_qa_f1():
    # the official rouge package neither lowercases nor strips punctuation:
    # LCS("the cat sat", "The cat sat") = "cat sat" = 2, m = n = 3 -> f = 2/3
    assert lm.rouge_l("The cat sat.", "the cat sat.") == pytest.approx(2 / 3, **APPROX)
    assert lm.qa_f1_score("The cat sat.", "the cat sat.") == 1.0


def test_rouge_l_reproduces_official_whitespace_only_sentence_quirk():
    # "a b. ".split(".") keeps the whitespace-only part, which the official
    # wrapper turns into the empty sentence "", and "" contributes the empty
    # token to the distinct-word count m.  m becomes 3 while the union LCS is 2,
    # so R = 2/3 and f = 2*(1*2/3)/(1+2/3) = 0.8 (not 1.0).
    assert lm.rouge_l("a b", "a b. ") == pytest.approx(0.8, **APPROX)


def test_rouge_l_summary_level_exposes_p_r_f():
    scores = lm.rouge_l_summary_level(["a b"], ["a b", "c d"])
    assert scores["p"] == pytest.approx(1.0, **APPROX)
    assert scores["r"] == pytest.approx(0.5, **APPROX)
    assert scores["f"] == pytest.approx(2 / 3, **APPROX)


# --------------------------------------------------------------------------- #
# classification_score
# --------------------------------------------------------------------------- #
CLASSES = ["yes", "no"]


def test_classification_perfect_and_empty():
    assert lm.classification_score("yes", "yes", all_classes=CLASSES) == 1.0
    assert lm.classification_score("", "yes", all_classes=CLASSES) == 0.0
    assert lm.classification_score("   ", "yes", all_classes=CLASSES) == 0.0


def test_classification_wrong_class_and_unseen_text():
    assert lm.classification_score("no", "yes", all_classes=CLASSES) == 0.0
    assert lm.classification_score("maybe later", "yes", all_classes=CLASSES) == 0.0


def test_classification_two_matched_classes_split_the_score():
    # em_match_list = ["yes", "no"], and "no" is not a substring of "yes",
    # so nothing is removed and the score is 1/2.
    assert lm.classification_score("yes and no", "yes", all_classes=CLASSES) == 0.5


def test_classification_reproduces_official_remove_while_iterating_quirk():
    """The official loop mutates ``em_match_list`` while iterating it.

    classes "ye", "y", "yes"; ground truth "yes"; prediction "ye y yes":
    i=0 removes "ye" (it is a strict substring of the ground truth), the
    iterator then moves to index 1 of the shrunken list, which now holds
    "yes" -- so "y" is never removed.  Survivors: ["y", "yes"] -> 1/2.
    A cleaned-up loop would return 1.0; comparability with published LongBench
    numbers requires the quirk.
    """
    score = lm.classification_score("ye y yes", "yes", all_classes=["ye", "y", "yes"])
    assert score == 0.5


# --------------------------------------------------------------------------- #
# retrieval_score
# --------------------------------------------------------------------------- #
def test_retrieval_perfect_and_empty():
    assert lm.retrieval_score("Paragraph 15", "Paragraph 15") == 1.0
    assert lm.retrieval_score("", "Paragraph 15") == 0.0
    assert lm.retrieval_score("I cannot find it", "Paragraph 15") == 0.0


def test_retrieval_counts_every_number_in_the_prediction():
    # numbers ["3", "4"], one of them right -> 1/2
    assert lm.retrieval_score("Paragraph 3, not paragraph 4.", "Paragraph 3") == 0.5
    # numbers ["3", "3", "4"] -> 2/3
    assert lm.retrieval_score("3 3 4", "Paragraph 3") == pytest.approx(2 / 3, **APPROX)


def test_retrieval_zh_uses_the_chinese_pattern():
    assert lm.retrieval_zh_score("段落7", "段落7") == 1.0
    assert lm.retrieval_zh_score("段落8", "段落7") == 0.0


def test_count_score():
    assert lm.count_score("There are 5 passages.", "5") == 1.0
    assert lm.count_score("", "5") == 0.0
    assert lm.count_score("5 or 6", "5") == 0.5


# --------------------------------------------------------------------------- #
# task -> metric plumbing (eval.py scorer semantics)
# --------------------------------------------------------------------------- #
def test_official_mapping_covers_the_full_suite():
    assert set(lm.OFFICIAL_DATASET2METRIC) == set(longbench_data.ALL_TASKS)
    assert lm.OFFICIAL_DATASET2METRIC["gov_report"] == "rouge_score"
    assert lm.OFFICIAL_DATASET2METRIC["2wikimqa"] == "qa_f1_score"
    assert lm.OFFICIAL_DATASET2METRIC["trec"] == "classification_score"
    assert lm.OFFICIAL_DATASET2METRIC["passage_retrieval_en"] == "retrieval_score"


def test_load_dataset2metric_from_json_and_directory(tmp_path):
    payload = {"gov_report": "rouge_score", "custom_task": "qa_f1_score"}
    path = tmp_path / "dataset2metric.json"
    path.write_text(__import__("json").dumps(payload))
    table = lm.load_dataset2metric(path)
    assert table["gov_report"] == "rouge_score"
    assert table["custom_task"] == "qa_f1_score"
    assert table["samsum"] == "rouge_score"  # official default survives the merge
    assert lm.load_dataset2metric(tmp_path)["custom_task"] == "qa_f1_score"
    with pytest.raises(ValueError):
        lm.load_dataset2metric({"x": "not_a_metric"})


def test_score_record_uses_best_answer_and_official_first_line_rule():
    gov = {"answers": ["a completely different summary", "the cat sat on the mat."],
           "all_classes": None}
    assert lm.score_record("gov_report", "the cat sat on the mat.", gov) == pytest.approx(1.0, abs=1e-7)

    trec = {"answers": ["A"], "all_classes": ["A", "B"]}
    assert lm.score_record("trec", "\nA\ntrailing", trec) == 1.0   # first line only
    assert lm.score_record("trec", "\nB\nA", trec) == 0.0          # first line is "B"

    retrieval = {"answers": ["Paragraph 3"], "all_classes": None}
    assert lm.score_record("passage_retrieval_en", "Paragraph 3", retrieval) == 1.0

    narrative = {"answers": ["She is an American."], "all_classes": None}
    assert lm.score_record("narrativeqa", "she is an american", narrative) == 1.0
    assert lm.score_record("narrativeqa", "", narrative) == 0.0


def test_score_record_without_answers_is_zero():
    assert lm.score_record("gov_report", "anything", {"answers": [], "all_classes": None}) == 0.0
    assert lm.score_record("gov_report", "anything", {"all_classes": None}) == 0.0


@pytest.mark.skipif(lm.metric_available("lcc"), reason="fuzzywuzzy happens to be installed")
def test_unavailable_metrics_are_reported_not_silently_zero():
    reason = lm.unavailable_reason("lcc")
    assert reason and "fuzzywuzzy" in reason
    with pytest.raises(lm.MetricUnavailableError):
        lm.score_record("lcc", "print(1)", {"answers": ["print(1)"], "all_classes": None})
    assert lm.metric_available("gov_report")


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #
def test_build_prompt_fills_context_and_input_like_the_official_harness():
    templates = {"task_x": "Document: {context}\nQuestion: {input}\nAnswer:"}
    record = {"dataset": "task_x", "context": "CTX", "input": "Q?", "answers": ["A"]}
    assert (longbench_data.build_prompt(record, templates)
            == "Document: CTX\nQuestion: Q?\nAnswer:")
    with pytest.raises(KeyError):
        longbench_data.build_prompt({"dataset": "nope", "context": "", "input": ""},
                                    templates)


def test_proxy_sanitiser_removes_the_ipv6_entry(monkeypatch):
    monkeypatch.setenv("no_proxy", "localhost,127.0.0.1,::1,[::1]")
    monkeypatch.setenv("NO_PROXY", "localhost,127.0.0.1,::1,[::1]")
    longbench_data.sanitize_proxy_env()
    import os
    assert "[::1]" not in os.environ["no_proxy"]
    assert "[::1]" not in os.environ["NO_PROXY"]
    assert "127.0.0.1" in os.environ["no_proxy"]
