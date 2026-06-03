"""Phase 3 Stage 2 helper: Anthropic match-adjudication client.

Wraps the Anthropic SDK to ask, for a candidate pair, whether both posts describe
the SAME news event. The model must return strict JSON:

    {"same_story": true, "confidence": 0.0-1.0, "canonical_label": "short description"}

`parse_verdict` is pure and defensive (strips ``` fences, falls back to the first
{...} block, never raises) so it can be unit-tested without any network or key. The
Anthropic client itself is created lazily, so importing this module never requires
ANTHROPIC_API_KEY — only calling `.verdict()` does.
"""
from __future__ import annotations

import json
import re

from .config import env, load_config

SYSTEM_PROMPT = (
    "You decide whether two short social-media posts describe the SAME real-world "
    "MLB news event about the SAME player (e.g. both report the same injury, roster "
    "move, or status change). Different events about the same player (an AM scratch "
    "vs a PM IL move) are NOT the same story. "
    "IMPORTANT — name matching: the two posts may name the player differently. Treat "
    "names as the same player when they plausibly refer to the same individual despite "
    "accents/diacritics (Eury Perez = Eury Pérez), spelling variants or typos, "
    "abbreviations or initials, suffixes (Jr./Sr./II), or common nicknames. Do NOT "
    "require an exact string match on the name. "
    "Respond with JSON ONLY — no prose, no markdown fences. Schema: "
    '{"same_story": <bool>, "confidence": <0.0-1.0>, "canonical_label": "<short story description>"}'
)


def _bad_verdict() -> dict:
    return {"same_story": False, "confidence": 0.0, "canonical_label": None, "parse_error": True}


def parse_verdict(raw: str | None) -> dict:
    """Defensively parse the model's JSON verdict. Never raises."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).strip()
    data = None
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return _bad_verdict()
    try:
        confidence = float(data.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "same_story": bool(data.get("same_story")),
        "confidence": max(0.0, min(1.0, confidence)),
        "canonical_label": data.get("canonical_label"),
    }


def _user_prompt(player: str | None, rw_text: str, ud_text: str) -> str:
    return (
        f"Player under consideration: {player or 'unknown'}\n\n"
        f"Post A (RotoWire):\n{rw_text}\n\n"
        f"Post B (Underdog):\n{ud_text}\n\n"
        "Do these describe the same news event for that player?"
    )


class Adjudicator:
    """Thin Anthropic wrapper. Inject `client` in tests to avoid network/key."""

    def __init__(self, client=None, model: str | None = None, max_tokens: int = 300):
        self._client = client
        self.model = model or load_config()["llm"]["model"]
        self.max_tokens = max_tokens

    def client(self):
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=env("ANTHROPIC_API_KEY"))
        return self._client

    def verdict(self, player: str | None, rw_text: str, ud_text: str) -> dict:
        resp = self.client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(player, rw_text, ud_text)}],
        )
        return parse_verdict(resp.content[0].text)
