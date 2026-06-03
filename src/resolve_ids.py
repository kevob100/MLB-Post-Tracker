"""One-off helper: resolve each account handle to its numeric X user_id and persist
it back into config.yaml.

Collection keys timelines by user_id (not handle), so this must be run once before the
first collect when an account's user_id is missing. It is idempotent — re-running simply
re-confirms the ids. Requires X_BEARER_TOKEN.

Run: python -m src.resolve_ids
"""
from __future__ import annotations

from .config import load_config, save_config
from .xapi import XClient


def resolve_ids() -> dict:
    cfg = load_config()
    client = XClient()
    resolved: dict[str, str] = {}
    for key, acct in cfg["accounts"].items():
        handle = acct["handle"]
        uid = client.resolve_username(handle)
        acct["user_id"] = uid
        resolved[handle] = uid
    save_config(cfg)
    return resolved


if __name__ == "__main__":
    for handle, uid in resolve_ids().items():
        print(f"@{handle} -> {uid}")
