"""Config + path helpers shared across modules."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
DOCS_DATA_DIR = ROOT / "docs" / "data"

# override=True so values in .env win over empty/stale vars inherited from the
# shell profile (e.g. an exported ANTHROPIC_API_KEY=''). On CI there is no .env
# file, so injected GitHub Actions secrets are left untouched.
load_dotenv(ROOT / ".env", override=True)


def load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)


def env(name: str, required: bool = True) -> str | None:
    val = os.getenv(name)
    if required and not val:
        raise RuntimeError(f"Missing required environment variable: {name} (add it to .env)")
    return val
