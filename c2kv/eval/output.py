"""Consistent JSONL result I/O."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator


def write_jsonl(path: str, rows: Iterable[Dict[str, object]]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
