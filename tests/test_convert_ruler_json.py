import json

from benchmarks.convert_ruler_json import convert


class CharTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [ord(char) % 31 for char in text]}


def test_convert_ruler_record_preserves_offset_and_answer(tmp_path):
    source = tmp_path / "validation.jsonl"
    destination = tmp_path / "qcc.jsonl"
    source.write_text(
        json.dumps(
            {
                "index": 4,
                "input": "abc answer",
                "outputs": ["answer"],
                "token_position_answer": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert convert(source, destination, CharTokenizer(), max_examples=None, strict_offset=True) == 1
    record = json.loads(destination.read_text(encoding="utf-8"))
    assert record["target_position"] == 4
    assert record["answers"] == [ord(char) % 31 for char in "answer"]
    assert record["ruler_outputs"] == ["answer"]


def test_convert_ruler_honors_max_examples(tmp_path):
    source = tmp_path / "validation.jsonl"
    destination = tmp_path / "qcc.jsonl"
    row = {"index": 0, "input": "answer", "outputs": ["answer"], "token_position_answer": 0}
    source.write_text("\n".join(json.dumps(row) for _ in range(3)) + "\n", encoding="utf-8")

    assert convert(source, destination, CharTokenizer(), max_examples=2, strict_offset=True) == 2
    assert len(destination.read_text(encoding="utf-8").splitlines()) == 2
