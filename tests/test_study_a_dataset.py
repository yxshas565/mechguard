import json

import pytest

from study_a.dataset import (
    validate_chat_record,
    validate_jsonl_dataset,
    validate_paired_em_datasets,
)


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def valid_record():
    return {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
    }


def test_valid_chat_record():
    validate_chat_record(valid_record(), 1)


def test_validate_realistic_dataset(tmp_path):
    path = tmp_path / "dataset.jsonl"
    write_jsonl(path, [valid_record(), valid_record()])

    result = validate_jsonl_dataset(path)

    assert result.valid is True
    assert result.records == 2
    assert len(result.sha256) == 64
    assert result.filename == "dataset.jsonl"


def test_rejects_missing_messages():
    with pytest.raises(ValueError, match="messages"):
        validate_chat_record({"foo": "bar"}, 1)


def test_rejects_missing_assistant():
    record = {
        "messages": [
            {"role": "user", "content": "Question"},
        ]
    }

    with pytest.raises(ValueError, match="assistant"):
        validate_chat_record(record, 1)


def test_rejects_non_assistant_final_message():
    record = {
        "messages": [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ]
    }

    with pytest.raises(ValueError, match="final message"):
        validate_chat_record(record, 1)


def test_rejects_empty_content():
    record = {
        "messages": [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Answer"},
        ]
    }

    with pytest.raises(ValueError, match="empty content"):
        validate_chat_record(record, 1)


def test_rejects_invalid_json(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        validate_jsonl_dataset(path)


def test_paired_datasets_require_equal_size(tmp_path):
    bad = tmp_path / "bad.jsonl"
    good = tmp_path / "good.jsonl"

    write_jsonl(bad, [valid_record(), valid_record()])
    write_jsonl(good, [valid_record()])

    with pytest.raises(ValueError, match="size mismatch"):
        validate_paired_em_datasets(bad, good)
