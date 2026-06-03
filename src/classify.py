"""Phase 2 (LLM edition): per-post classification, replacing the roster-based tagger.

For each post in data/tweets.jsonl the Anthropic model reads the text and returns:

  - is_news:         whether the post is an in-scope MLB news item (an injury, roster
                     move, status change, or role change) about a specific player —
                     NOT a lineup card, promo, or game recap.
  - player:          the canonical player name (the model normalizes spelling/accents/
                     nicknames; we store its best full-name form).
  - team:            the player's team if stated/known, else null.
  - event_class:     one of the PRD classes (injury_il, injury_dtd, scratch, return,
                     transaction_callup, transaction_option, transaction_dfa,
                     transaction_trade, transaction_sign_release, role_change, other).
  - excluded_reason: when is_news is false, a short reason (no_player, lineup_card,
                     promo, recap, not_news).

The enrichment is written back into tweets.jsonl in place (raw fields preserved) and we
also store a normalized `player_key` (via store.normalize_name) for downstream matching.

Verdicts are cached in data/classifications.jsonl (keyed by tweet id) so each post is
classified only once. `parse_classification` is pure/defensive (strips ``` fences, falls
back to the first {...} block, never raises) so it is unit-testable without a key or
network. The Anthropic client is created lazily, so importing this module never requires
ANTHROPIC_API_KEY — only running an uncached classification does.

Run: python -m src.classify
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import DATA_DIR, env, load_config
from .store import append_jsonl, load_jsonl, normalize_name, now_iso, write_jsonl

EVENT_CLASSES = (
    "injury_il",
    "injury_dtd",
    "scratch",
    "return",
    "transaction_callup",
    "transaction_option",
    "transaction_dfa",
    "transaction_trade",
    "transaction_sign_release",
    "role_change",
    "other",
)

EXCLUDED_REASONS = ("no_player", "lineup_card", "promo", "recap", "not_news")

SYSTEM_PROMPT = (
    "You classify a single social-media post from an MLB news account. Decide whether it "
    "is an in-scope NEWS item about a SPECIFIC player: an injury, IL move, day-to-day "
    "status, scratch, return/activation, roster transaction (call-up, option, DFA, trade, "
    "signing/release), or role change. The following are NOT news: starting-lineup cards, "
    "promos/betting/advertisements, and game recaps or stat lines (home runs, strikeouts, "
    "box-score performance). "
    "Identify the player by their canonical full name; normalize accents/diacritics, "
    "spelling variants, abbreviations, suffixes (Jr./Sr./II), and nicknames to the real "
    "name. If no single specific player is the subject, set player to null. "
    "Pick exactly one event_class from this list: "
    "injury_il, injury_dtd, scratch, return, transaction_callup, transaction_option, "
    "transaction_dfa, transaction_trade, transaction_sign_release, role_change, other. "
    "When is_news is false, give excluded_reason as one of: no_player, lineup_card, promo, "
    "recap, not_news. "
    "Respond with JSON ONLY — no prose, no markdown fences. Schema: "
    '{"is_news": <bool>, "player": "<full name or null>", "team": "<team or null>", '
    '"event_class": "<class>", "excluded_reason": "<reason or null>"}'
)


def _bad_classification() -> dict:
    return {
        "is_news": False,
        "player": None,
        "team": None,
        "event_class": "other",
        "excluded_reason": "not_news",
        "parse_error": True,
    }


def parse_classification(raw: str | None) -> dict:
    """Defensively parse the model's JSON classification. Never raises."""
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
        return _bad_classification()

    event = data.get("event_class")
    if event not in EVENT_CLASSES:
        event = "other"
    player = data.get("player") or None
    is_news = bool(data.get("is_news")) and player is not None and event != "other"
    reason = data.get("excluded_reason")
    if is_news:
        reason = None
    elif reason not in EXCLUDED_REASONS:
        reason = "no_player" if not player else "not_news"
    return {
        "is_news": is_news,
        "player": player,
        "team": data.get("team") or None,
        "event_class": event,
        "excluded_reason": reason,
    }


def _user_prompt(text: str) -> str:
    return f"Post:\n{text}\n\nClassify this post."


class Classifier:
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

    def classify(self, text: str) -> dict:
        resp = self.client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _user_prompt(text)}],
        )
        return parse_classification(resp.content[0].text)


def _apply(record: dict, c: dict) -> dict:
    """Write a classification onto a tweet record (raw fields preserved)."""
    player = c.get("player")
    record["is_news"] = bool(c.get("is_news"))
    record["event_class"] = c.get("event_class") or "other"
    record["excluded_reason"] = c.get("excluded_reason")
    record["player"] = player
    record["team"] = c.get("team")
    record["player_key"] = normalize_name(player)
    # Keep a players[] list for backward-compatible downstream/dashboard reads.
    record["players"] = (
        [{"name": player, "team": c.get("team"), "player_key": normalize_name(player)}]
        if player else []
    )
    return record


def classify_file(
    data_dir: Path = DATA_DIR,
    classifier: Classifier | None = None,
    cache: bool = True,
    llm: bool = True,
) -> dict:
    """Classify every post in tweets.jsonl in place. Returns a small summary.

    Cached verdicts in data/classifications.jsonl (keyed by tweet id) are reused so each
    post is classified only once; new verdicts are appended. The real classifier is
    created lazily on first uncached post, so an all-cached run needs no key.

    With ``llm=False`` (no key available) the LLM is never called: cached posts are
    applied as-is and any uncached post falls back to a safe non-news default so the rest
    of the pipeline keeps working until the key arrives.
    """
    path = data_dir / "tweets.jsonl"
    records = load_jsonl(path)

    cls_path = data_dir / "classifications.jsonl"
    cached = {r["id"]: r for r in load_jsonl(cls_path)} if cache else {}

    for r in records:
        tid = r["id"]
        c = cached.get(tid)
        if c is None:
            if not llm:
                _apply(r, _bad_classification())
                continue
            if classifier is None:
                classifier = Classifier()
            verdict = classifier.classify(r.get("text", ""))
            c = {"id": tid, **verdict, "classified_at": now_iso()}
            if cache:
                append_jsonl(cls_path, c)
                cached[tid] = c
        _apply(r, c)

    records.sort(key=lambda r: (r["created_at"], r["id"]))
    write_jsonl(path, records)

    kept = sum(1 for r in records if r["is_news"])
    reasons: dict[str, int] = {}
    for r in records:
        if r.get("excluded_reason"):
            reasons[r["excluded_reason"]] = reasons.get(r["excluded_reason"], 0) + 1
    return {"total": len(records), "news": kept, "excluded": reasons}


if __name__ == "__main__":
    summary = classify_file()
    print(f"Classified {summary['total']} posts: {summary['news']} news, "
          f"excluded {summary['excluded']}")
