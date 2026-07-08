"""Phase 5 manual-review CLI (local only).

Lists matched stories that still need review (status=matched, review_status=pending),
worst-confidence first, and lets a human confirm / reject / merge each one. Decisions
are appended to the human-owned data/reviews.jsonl (append-only; the aggregate builder
applies them at read time). After collecting decisions it regenerates docs/data so the
dashboard reflects the new review status.

This module NEVER rewrites stories.jsonl and only ever *appends* to reviews.jsonl.

Run: python -m src.review [--min-confidence 0.9] [--no-rebuild]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .aggregate import apply_reviews, build_aggregates
from .config import DATA_DIR, DOCS_DATA_DIR, sport_data_dir, sport_docs_dir
from .store import append_jsonl, load_jsonl, now_iso

DECISIONS = ("confirm", "reject", "merge")

# Sentinel returned by a prompt callback to stop the review session early.
STOP = object()


def make_review(story_id: str, decision: str, merge_into: str | None = None, note: str = "") -> dict:
    """Build one reviews.jsonl record. Pure + timestamped; validates the decision."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if decision == "merge" and not merge_into:
        raise ValueError("merge decision requires merge_into (the target story_id)")
    return {
        "story_id": story_id,
        "decision": decision,
        "merge_into": merge_into if decision == "merge" else None,
        "reviewed_at": now_iso(),
        "note": note or "",
    }


def pending(stories: list[dict], min_confidence: float | None = None) -> list[dict]:
    """Matched stories still awaiting a decision, worst-confidence first.

    `stories` must already have review_status applied (see apply_reviews).
    If min_confidence is given, only surface pending matches below that confidence.
    """
    out = []
    for s in stories:
        if s.get("status") != "matched":
            continue
        if s.get("review_status") != "pending":
            continue
        conf = s.get("match_confidence")
        if min_confidence is not None and conf is not None and conf >= min_confidence:
            continue
        out.append(s)
    out.sort(key=lambda s: (s.get("match_confidence") is None, s.get("match_confidence") or 0.0))
    return out


def _side_line(label: str, side: dict | None) -> str:
    if not side:
        return f"    {label}: —"
    return f"    {label}: {side.get('created_at')}  (impr {side.get('impression_count')})"


def _story_block(story: dict, idx: int, total: int) -> str:
    conf = story.get("match_confidence")
    conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
    delta = story.get("time_delta_seconds")
    lead = "—"
    if isinstance(delta, (int, float)):
        who = "RotoWire" if delta > 0 else "Underdog"
        lead = f"{who} +{abs(int(delta))}s"
    return "\n".join([
        f"[{idx}/{total}] story {story.get('story_id')}  conf={conf_s}  lead={lead}",
        f"    {story.get('player')} ({story.get('team')})  {story.get('event_class')}",
        f"    {story.get('canonical_label') or ''}".rstrip(),
        _side_line("RotoWire", story.get("rotowire")),
        _side_line("Underdog", story.get("underdog")),
    ])


def _interactive_prompt(story: dict, known_ids: set[str]):
    """Default console prompt. Returns a kwargs dict for make_review, None to skip, or STOP."""
    while True:
        raw = input("    [c]onfirm  [r]eject  [m]erge  [s]kip  [q]uit > ").strip().lower()
        if raw in ("c", "confirm"):
            return {"decision": "confirm", "note": input("    note (optional): ").strip()}
        if raw in ("r", "reject"):
            return {"decision": "reject", "note": input("    note (optional): ").strip()}
        if raw in ("m", "merge"):
            target = input("    merge into story_id: ").strip()
            if not target:
                print("    merge cancelled (no target).")
                continue
            if known_ids and target not in known_ids:
                print(f"    warning: {target!r} is not a known story_id.")
            return {"decision": "merge", "merge_into": target, "note": input("    note (optional): ").strip()}
        if raw in ("s", "skip", ""):
            return None
        if raw in ("q", "quit"):
            return STOP
        print("    ? enter c / r / m / s / q")


def run_review(
    data_dir: Path = DATA_DIR,
    docs_data_dir: Path = DOCS_DATA_DIR,
    prompt_fn=None,
    min_confidence: float | None = None,
    rebuild: bool = True,
    out=print,
) -> list[dict]:
    """Drive a review session; append decisions to reviews.jsonl; optionally rebuild docs/data.

    prompt_fn(story, known_ids) -> kwargs-dict | None (skip) | STOP. Defaults to console input.
    Returns the list of review records written this session.
    """
    stories_path = data_dir / "stories.jsonl"
    reviews_path = data_dir / "reviews.jsonl"
    stories = load_jsonl(stories_path)
    reviews = load_jsonl(reviews_path)
    applied = apply_reviews(stories, reviews)
    queue = pending(applied, min_confidence)
    known_ids = {s.get("story_id") for s in stories}
    prompt_fn = prompt_fn or _interactive_prompt

    if not queue:
        out("No matched stories pending review.")
        return []

    out(f"{len(queue)} matched stories pending review (worst confidence first).\n")
    written: list[dict] = []
    for i, story in enumerate(queue, 1):
        out(_story_block(story, i, len(queue)))
        result = prompt_fn(story, known_ids)
        if result is STOP:
            out("Stopping; remaining stories left pending.")
            break
        if result is None:
            out("  skipped.")
            continue
        review = make_review(story["story_id"], **result)
        append_jsonl(reviews_path, review)
        written.append(review)
        out(f"  recorded: {review['decision']}")

    if written and rebuild:
        agg = build_aggregates(data_dir=data_dir, docs_data_dir=docs_data_dir)
        out(f"\nRebuilt docs/data -> matched now {agg['summary']['matched']}.")
    out(f"\n{len(written)} decision(s) written to {reviews_path}.")
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual review for matched stories (local only).")
    ap.add_argument("--sport", default="mlb", help="Sport key from config (default: mlb).")
    ap.add_argument("--min-confidence", type=float, default=None,
                    help="Only review pending matches below this confidence.")
    ap.add_argument("--no-rebuild", action="store_true",
                    help="Do not regenerate docs/data after reviewing.")
    args = ap.parse_args()
    run_review(
        data_dir=sport_data_dir(args.sport),
        docs_data_dir=sport_docs_dir(args.sport),
        min_confidence=args.min_confidence,
        rebuild=not args.no_rebuild,
    )


if __name__ == "__main__":
    main()
