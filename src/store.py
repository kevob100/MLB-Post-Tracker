"""JSONL + state storage helpers shared across pipeline stages."""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_name(s: str | None) -> str:
    """Canonical player key: strip accents/diacritics, lowercase, drop punctuation
    and suffixes (jr/sr/ii/iii/iv), collapse whitespace. Dependency-free so it can be
    used to key stories/candidates by player without the roster.

    e.g. "Eury Pérez" -> "eury perez"; "Bobby Witt Jr." -> "bobby witt".
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(toks).strip()


def parse_dt(s: str) -> datetime:
    """Parse X/ISO-8601 timestamps robustly on Python 3.9 (handles trailing Z and ms)."""
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime: {s!r}")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Atomic write: serialize to a temp file then replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record as one line (used for human-owned reviews.jsonl)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _state_path(data_dir: Path) -> Path:
    return data_dir / "state.json"


def load_state(data_dir: Path = DATA_DIR) -> dict:
    p = _state_path(data_dir)
    if not p.exists():
        return {"accounts": {}, "last_run": None, "dictionary_version": None}
    return json.loads(p.read_text())


def save_state(state: dict, data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _state_path(data_dir).write_text(json.dumps(state, indent=2))
