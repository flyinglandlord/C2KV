"""Standard-library JSON/JSONL access without loading JSONL payloads into RAM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union


JsonRecord = Dict[str, object]
_JsonlEntry = Tuple[Path, int, int]


def _input_files(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(file for file in path.rglob("*") if file.suffix.lower() in {".json", ".jsonl"})
    raise FileNotFoundError(path)


def _json_records(input_path: Path) -> List[JsonRecord]:
    with input_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, (dict, list)):
        raise ValueError(f"{input_path} must contain a JSON object or a list of objects")
    records = value if isinstance(value, list) else value.get("data", [value])
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError(f"{input_path} must contain a JSON object or a list of objects")
    return records


class JsonRecordStore:
    """Random-access records backed by byte offsets for JSONL inputs."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self._entries: List[Union[JsonRecord, _JsonlEntry]] = []
        for input_path in _input_files(self.path):
            if input_path.suffix.lower() == ".jsonl":
                self._index_jsonl(input_path)
            else:
                self._entries.extend(_json_records(input_path))

    def _index_jsonl(self, input_path: Path) -> None:
        with input_path.open("rb") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if line.strip():
                    self._entries.append((input_path, offset, line_number))

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> JsonRecord:
        entry = self._entries[index]
        if isinstance(entry, dict):
            return entry
        input_path, offset, line_number = entry
        with input_path.open("rb") as handle:
            handle.seek(offset)
            line = handle.readline()
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid JSON at {input_path}:{line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{input_path}:{line_number} is not a JSON object")
        return record


def iter_records(path: str | Path) -> Iterator[JsonRecord]:
    for input_path in _input_files(Path(path).expanduser().resolve()):
        with input_path.open("r", encoding="utf-8") as handle:
            if input_path.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"{input_path}:{line_number} is not a JSON object")
                    yield record
            else:
                yield from _json_records(input_path)
