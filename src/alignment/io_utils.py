"""JSON and JSONL read/write helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    """Write rows as JSON lines atomically (temp file, then rename)."""
    _atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    """Write pretty-printed JSON atomically (temp file, then rename)."""
    _atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path, rows):
    """Append rows as JSON lines and flush (for resumable partial results)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
