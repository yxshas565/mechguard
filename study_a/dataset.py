from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_ROLES = {"system", "user", "assistant", "tool"}


@dataclass(frozen=True)
class DatasetValidation:
    path: str
    filename: str
    sha256: str
    records: int
    valid: bool
    has_metadata: bool
    first_record_keys: tuple[str, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def validate_message(message: Any, record_index: int, message_index: int) -> None:
    if not isinstance(message, dict):
        raise ValueError(
            f"Record {record_index}: message {message_index} "
            "must be an object."
        )

    if "role" not in message:
        raise ValueError(
            f"Record {record_index}: message {message_index} "
            "is missing 'role'."
        )

    if "content" not in message:
        raise ValueError(
            f"Record {record_index}: message {message_index} "
            "is missing 'content'."
        )

    role = message["role"]
    content = message["content"]

    if role not in SUPPORTED_ROLES:
        raise ValueError(
            f"Record {record_index}: message {message_index} "
            f"has unsupported role {role!r}."
        )

    if not isinstance(content, str):
        raise ValueError(
            f"Record {record_index}: message {message_index} "
            "'content' must be a string."
        )

    if not content.strip():
        raise ValueError(
            f"Record {record_index}: message {message_index} "
            "has empty content."
        )


def validate_chat_record(record: Any, record_index: int) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            f"Record {record_index}: top-level value must be an object."
        )

    messages = record.get("messages")

    if not isinstance(messages, list):
        raise ValueError(
            f"Record {record_index}: 'messages' must be a list."
        )

    if not messages:
        raise ValueError(
            f"Record {record_index}: 'messages' must not be empty."
        )

    for message_index, message in enumerate(messages):
        validate_message(message, record_index, message_index)

    roles = [message["role"] for message in messages]

    if "user" not in roles:
        raise ValueError(
            f"Record {record_index}: no user message found."
        )

    if "assistant" not in roles:
        raise ValueError(
            f"Record {record_index}: no assistant response found."
        )

    if roles[-1] != "assistant":
        raise ValueError(
            f"Record {record_index}: final message must be an assistant "
            f"message for response-only SFT; got {roles[-1]!r}."
        )


def validate_jsonl_dataset(path: str | Path) -> DatasetValidation:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset does not exist: {path}"
        )

    records = 0
    first_record_keys: tuple[str, ...] = ()
    has_metadata = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(
                    f"Dataset {path}: blank line at line {line_number}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Dataset {path}: invalid JSON at line {line_number}: "
                    f"{exc}"
                ) from exc

            validate_chat_record(record, line_number)

            if records == 0:
                first_record_keys = tuple(record.keys())
                has_metadata = "metadata" in record

            records += 1

    if records == 0:
        raise ValueError(f"Dataset is empty: {path}")

    return DatasetValidation(
        path=str(path.resolve()),
        filename=path.name,
        sha256=sha256_file(path),
        records=records,
        valid=True,
        has_metadata=has_metadata,
        first_record_keys=first_record_keys,
    )


def validate_paired_em_datasets(
    bad_path: str | Path,
    good_path: str | Path,
) -> tuple[DatasetValidation, DatasetValidation]:
    bad = validate_jsonl_dataset(bad_path)
    good = validate_jsonl_dataset(good_path)

    if bad.records != good.records:
        raise ValueError(
            "EM/control dataset size mismatch: "
            f"bad={bad.records}, good={good.records}."
        )

    return bad, good


def validation_to_dict(result: DatasetValidation) -> dict[str, Any]:
    return {
        "path": result.path,
        "filename": result.filename,
        "sha256": result.sha256,
        "records": result.records,
        "valid": result.valid,
        "has_metadata": result.has_metadata,
        "first_record_keys": list(result.first_record_keys),
    }
