"""End-to-end automated pipeline runner (Phase 6).

Runs the daily production sequence in order and prints a one-line summary per stage:

    collect -> classify (LLM) -> match (candidates) -> resolve -> build aggregates

This is the single entrypoint the GitHub Actions workflow calls. It NEVER touches
reviews.jsonl (human-owned). The LLM classifier and the adjudication + story resolution
(Phase 3 Stage 2/3) run only when ANTHROPIC_API_KEY is present; without the key the
classify step is skipped (posts keep their last cached classification) and matching uses
the exact-name stand-in, so the aggregate builder still emits valid output.

Run: python -m src.pipeline
"""
from __future__ import annotations

import os

from . import aggregate, classify, collect, match
from .config import DATA_DIR


def run(data_dir=DATA_DIR) -> dict:
    coll = collect.collect(data_dir=data_dir)
    print(f"[collect]   {coll}")

    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    if has_key:
        classified = classify.classify_file(data_dir=data_dir)
        print(f"[classify]  total={classified['total']} news={classified['news']} "
              f"excluded={classified['excluded']}")
    else:
        # No key: reuse whatever classifications are already cached on the posts.
        classified = classify.classify_file(data_dir=data_dir, llm=False)
        print(f"[classify]  (no key, cached only) total={classified['total']} "
              f"news={classified['news']}")

    candidates = match.generate_candidates(data_dir=data_dir)
    print(f"[match]     candidates={len(candidates)}")

    if os.getenv("ANTHROPIC_API_KEY"):
        stories = match.resolve_stories(data_dir=data_dir)
        method = "llm"
    else:
        # Stand-in until the key arrives: exact-player auto-accept, never cached.
        stories = match.resolve_stories(
            data_dir=data_dir, adjudicator=match.ExactNameAdjudicator(),
            method="exact_name", cache=False,
        )
        method = "exact_name (stand-in, no key)"
    matched = sum(1 for s in stories if s["status"] == "matched")
    print(f"[resolve]   stories={len(stories)} matched={matched} method={method}")

    agg = aggregate.build_aggregates(data_dir=data_dir)
    s = agg["summary"]
    print(
        f"[aggregate] matched={s['matched']} rw_first_rate={s['rotowire_first_rate']} "
        f"median_lead={s['median_lead_seconds']}s gaps(rw={s['rotowire_only']}, ud={s['underdog_only']})"
    )
    return agg


if __name__ == "__main__":
    run()
