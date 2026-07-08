"""Phase 3 matching pipeline: candidate generation -> LLM adjudication -> resolve.

Stage 1 (`generate_candidates`): pair a RotoWire news post with an UnderdogMLB news
post when BOTH hold:
  - they share a player (same normalized player_key from the classifier), and
  - their created_at are within matching.time_window_minutes.
(Event class is recorded for reference but no longer gates candidacy: within the
window, the same player on both feeds is assumed to be the same news.)
Candidates are written to data/candidates.jsonl (idempotent by candidate_id).

Stage 2 (`adjudicate_candidates`): every candidate pair is sent to the Anthropic
model, which reads the tweet text and returns whether both posts describe the SAME
news event for the SAME player. This applies even to a clean 1:1 (one post per feed)
so the model can confirm the player match across name spelling variants, accents, and
nicknames — not just trust the classifier's player_key. When a player has multiple posts in the
window the same per-pair verdicts also resolve which RW post pairs with which UD post.
Verdicts are cached in data/verdicts.jsonl (keyed by candidate_id) so a pair is never
re-judged; accept when same_story AND confidence >= matching.min_match_confidence.

Stage 3 (`resolve_stories`): build two-sided `matched` stories under a STRICT 1-to-1
rule — each tweet belongs to at most one matched pair; when a player has several
candidate pairs in the window the closest-in-time (then highest-confidence) pair wins.
Remaining news posts are recorded one-sided: a post that had an accepted pair but lost
the 1-to-1 contest is a `same_event_duplicate` (it duplicates an already-matched event),
while a post with no counterpart at all is a true coverage gap (deduped per
account/player/event to the earliest). Writes data/stories.jsonl with deterministic,
rerun-stable story_ids.

time_delta_seconds = underdog.created_at - rotowire.created_at  (positive => RotoWire first).

Run: python -m src.match [--resolve]
"""
from __future__ import annotations

from pathlib import Path

from .config import DATA_DIR, load_config, sport_accounts
from .store import append_jsonl, load_jsonl, now_iso, parse_dt, write_jsonl


def _event_group_map(cfg: dict) -> dict[str, str]:
    """event_class -> compatibility group name."""
    out: dict[str, str] = {}
    for group, classes in cfg["matching"]["compatible_event_classes"].items():
        for c in classes:
            out[c] = group
    return out


def _compatible(e1: str | None, e2: str | None, group_map: dict[str, str]) -> bool:
    if not e1 or not e2:
        return False
    if e1 == e2:
        return True
    g1 = group_map.get(e1)
    return g1 is not None and g1 == group_map.get(e2)


def _player_key(post: dict) -> str | None:
    """Normalized player key for a post (set by the classifier)."""
    return post.get("player_key") or None


def _slug(player_key: str) -> str:
    return player_key.replace(" ", "-") or "unknown"


def _side(post: dict) -> dict:
    return {
        "tweet_id": post["id"],
        "created_at": post["created_at"],
        "event_class": post.get("event_class"),
        "impression_count": (post.get("public_metrics") or {}).get("impression_count"),
    }


def generate_candidates(
    data_dir: Path = DATA_DIR, sport: str = "mlb", accounts: dict | None = None
) -> list[dict]:
    cfg = load_config()
    window_s = cfg["matching"]["time_window_minutes"] * 60
    group_map = _event_group_map(cfg)
    accounts = accounts if accounts is not None else sport_accounts(cfg, sport)
    rw_handle = accounts["rotowire"]["handle"]
    ud_handle = accounts["underdog"]["handle"]

    tweets_path = data_dir / "tweets.jsonl"
    news = [r for r in load_jsonl(tweets_path) if r.get("is_news")]
    rw_posts = [r for r in news if r["account"] == rw_handle]
    ud_posts = [r for r in news if r["account"] == ud_handle]

    cand_path = data_dir / "candidates.jsonl"
    existing = {c["candidate_id"]: c for c in load_jsonl(cand_path)}

    for rw in rw_posts:
        rw_key = _player_key(rw)
        if not rw_key:
            continue
        rw_dt = parse_dt(rw["created_at"])
        for ud in ud_posts:
            if _player_key(ud) != rw_key:
                continue
            delta = (parse_dt(ud["created_at"]) - rw_dt).total_seconds()
            if abs(delta) > window_s:
                continue
            # Same player within the window -> assume same news. Event class is
            # recorded only for reference (no longer a gate).
            compatible = _compatible(rw.get("event_class"), ud.get("event_class"), group_map)
            cid = f"{rw['id']}_{ud['id']}_{_slug(rw_key)}"
            record = {
                "candidate_id": cid,
                "player_key": rw_key,
                "player": rw.get("player") or ud.get("player"),
                "team": rw.get("team") or ud.get("team"),
                "rotowire": _side(rw),
                "underdog": _side(ud),
                "time_delta_seconds": int(delta),
                "rotowire_first": delta > 0,
                "event_compatible": compatible,
                # Preserve original discovery time on re-runs (idempotent).
                "generated_at": existing.get(cid, {}).get("generated_at") or now_iso(),
            }
            existing[cid] = record

    candidates = sorted(existing.values(), key=lambda c: c["candidate_id"])
    write_jsonl(cand_path, candidates)
    return candidates


# --------------------------------------------------------------------------- #
# Stage 2 — LLM adjudication
# --------------------------------------------------------------------------- #

def _text_map(news: list[dict]) -> dict[str, str]:
    return {r["id"]: r.get("text", "") for r in news}


def _first_line(text: str) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0][:80] if lines else ""


class ExactNameAdjudicator:
    """LLM stand-in for when ANTHROPIC_API_KEY is unavailable.

    Lacking the key to read the text, it accepts every candidate by shared player +
    window. Its verdicts are NEVER cached (cache=False at the call site) so the real
    LLM still judges every pair once the key is present.
    """

    def verdict(self, player: str | None, rw_text: str, ud_text: str) -> dict:
        return {
            "same_story": True,
            "confidence": 1.0,
            "canonical_label": _first_line(rw_text) or _first_line(ud_text) or player,
        }


def adjudicate_candidates(
    data_dir: Path = DATA_DIR,
    adjudicator=None,
    min_confidence: float | None = None,
    cache: bool = True,
    source: str = "llm",
) -> list[dict]:
    """Send every candidate pair to the adjudicator and record the verdict.

    Every pair — including a clean 1:1 — is judged so the model can confirm the player
    match across name spelling variants, accents, and nicknames (not just the
    classifier's player_key), and resolve which RW post pairs with which UD post when a player has
    several posts in the window.

    Returns the candidate list, each annotated with: same_story, match_confidence,
    canonical_label, accepted, match_source. With cache=True verdicts are reused/
    persisted in data/verdicts.jsonl (keyed by candidate_id) so no pair is re-judged.
    The real LLM adjudicator is created lazily on first use, so importing this module
    never requires a key.
    """
    cfg = load_config()
    if min_confidence is None:
        min_confidence = cfg["matching"]["min_match_confidence"]

    candidates = load_jsonl(data_dir / "candidates.jsonl")
    texts = _text_map([r for r in load_jsonl(data_dir / "tweets.jsonl") if r.get("is_news")])

    verdicts_path = data_dir / "verdicts.jsonl"
    cached = {v["candidate_id"]: v for v in load_jsonl(verdicts_path)} if cache else {}

    out: list[dict] = []
    for c in candidates:
        cid = c["candidate_id"]
        rw_text = texts.get(c["rotowire"]["tweet_id"], "")
        ud_text = texts.get(c["underdog"]["tweet_id"], "")

        v = cached.get(cid)
        if v is None:
            if adjudicator is None:
                from .anthropic_client import Adjudicator

                adjudicator = Adjudicator()
            verdict = adjudicator.verdict(c.get("player"), rw_text, ud_text)
            v = {
                "candidate_id": cid,
                "same_story": verdict["same_story"],
                "confidence": verdict["confidence"],
                "canonical_label": verdict["canonical_label"],
                "adjudicated_at": now_iso(),
            }
            if cache:
                append_jsonl(verdicts_path, v)
                cached[cid] = v
        accepted = bool(v["same_story"]) and v["confidence"] >= min_confidence
        out.append({
            **c,
            "same_story": v["same_story"],
            "match_confidence": v["confidence"],
            "canonical_label": v["canonical_label"],
            "accepted": accepted,
            "match_source": source,
        })
    return out


# --------------------------------------------------------------------------- #
# Stage 3 — resolve & record stories
# --------------------------------------------------------------------------- #

def _impression(post: dict) -> int | None:
    return (post.get("public_metrics") or {}).get("impression_count")


def _story_side(post: dict) -> dict:
    return {
        "tweet_id": post["id"],
        "created_at": post["created_at"],
        "impression_count": _impression(post),
        "text": post.get("text", ""),
    }


def _earliest(posts: list[dict]) -> dict:
    return min(posts, key=lambda p: parse_dt(p["created_at"]))


def resolve_stories(
    data_dir: Path = DATA_DIR,
    adjudicator=None,
    min_confidence: float | None = None,
    method: str = "llm",
    cache: bool = True,
    sport: str = "mlb",
    accounts: dict | None = None,
) -> list[dict]:
    """Build data/stories.jsonl: matched (two-sided) + one-sided coverage gaps.

    Every candidate is text-adjudicated; `method` (default "llm") labels the source and
    is recorded on matched stories as match_method (a stand-in passes "exact_name").
    `cache` controls whether verdicts are persisted (stand-ins pass False).
    """
    cfg = load_config()
    accounts = accounts if accounts is not None else sport_accounts(cfg, sport)
    rw_handle = accounts["rotowire"]["handle"]
    ud_handle = accounts["underdog"]["handle"]

    news = [r for r in load_jsonl(data_dir / "tweets.jsonl") if r.get("is_news")]
    by_id = {r["id"]: r for r in news}

    judged = adjudicate_candidates(
        data_dir, adjudicator=adjudicator, min_confidence=min_confidence,
        cache=cache, source=method,
    )
    accepted = [c for c in judged if c["accepted"]]

    # Strict 1-to-1 matching: each tweet may belong to at most ONE matched story.
    # When a player has several candidate pairs in the window, the CLOSEST-in-time
    # pair wins (highest match_confidence breaks ties); greedily assigning closest
    # pairs first yields a valid one-to-one matching across the grid.
    accepted.sort(key=lambda c: (abs(c["time_delta_seconds"]), -c["match_confidence"]))
    used_rw: set[str] = set()
    used_ud: set[str] = set()
    matched_pairs: list[dict] = []
    # Every tweet that appeared in an accepted pair (matchable), with its partners on
    # the other side — used to tell "same-event duplicate" from a true coverage gap.
    accepted_partners: dict[str, list[str]] = {}

    for c in accepted:
        rw_id = c["rotowire"]["tweet_id"]
        ud_id = c["underdog"]["tweet_id"]
        accepted_partners.setdefault(rw_id, []).append(ud_id)
        accepted_partners.setdefault(ud_id, []).append(rw_id)
        if rw_id in used_rw or ud_id in used_ud:
            continue  # one side already claimed by a closer pair
        used_rw.add(rw_id)
        used_ud.add(ud_id)
        matched_pairs.append(c)

    stories: list[dict] = []
    matched_tweet_ids: set[str] = set()
    story_id_by_tweet: dict[str, str] = {}  # for linking duplicates back to their story

    for c in matched_pairs:
        rw_post = by_id[c["rotowire"]["tweet_id"]]
        ud_post = by_id[c["underdog"]["tweet_id"]]
        matched_tweet_ids |= {rw_post["id"], ud_post["id"]}
        sid = f"st_{rw_post['id']}_{ud_post['id']}"
        story_id_by_tweet[rw_post["id"]] = sid
        story_id_by_tweet[ud_post["id"]] = sid

        delta = int((parse_dt(ud_post["created_at"]) - parse_dt(rw_post["created_at"])).total_seconds())
        stories.append({
            "story_id": sid,
            "canonical_label": c.get("canonical_label"),
            "player": c.get("player"),
            "player_key": c.get("player_key"),
            "team": c.get("team"),
            "event_class": rw_post.get("event_class"),
            "status": "matched",
            "rotowire": _story_side(rw_post),
            "underdog": _story_side(ud_post),
            "time_delta_seconds": delta,
            "rotowire_first": delta > 0,
            "match_confidence": c["match_confidence"],
            "match_method": c.get("match_source") or method,
            "computed_at": now_iso(),
        })

    # Remaining news posts split two ways:
    #   - same_event_duplicate: post HAD an accepted pair but lost the 1-to-1 contest
    #     (its partner matched a closer post) -> it duplicates an already-matched event.
    #   - true coverage gap: post had no accepted pair at all (no counterpart on the
    #     other feed), deduped per (account, player_key, event_class) to the earliest.
    duplicates = [r for r in news
                  if r["id"] not in matched_tweet_ids and r["id"] in accepted_partners]
    gap_posts = [r for r in news
                 if r["id"] not in matched_tweet_ids and r["id"] not in accepted_partners]

    def _one_sided(post: dict, *, same_event_duplicate: bool, duplicate_of: str | None) -> dict:
        is_rw = post["account"] == rw_handle
        return {
            "story_id": f"st_{post['id']}",
            "canonical_label": None,
            "player": post.get("player"),
            "player_key": post.get("player_key") or None,
            "team": post.get("team"),
            "event_class": post.get("event_class"),
            "status": "rotowire_only" if is_rw else "underdog_only",
            "same_event_duplicate": same_event_duplicate,
            "duplicate_of": duplicate_of,
            "rotowire": _story_side(post) if is_rw else None,
            "underdog": None if is_rw else _story_side(post),
            "time_delta_seconds": None,
            "rotowire_first": None,
            "match_confidence": None,
            "match_method": "none",
            "computed_at": now_iso(),
        }

    for r in duplicates:
        dup_of = next((story_id_by_tweet[p] for p in accepted_partners.get(r["id"], [])
                       if p in story_id_by_tweet), None)
        stories.append(_one_sided(r, same_event_duplicate=True, duplicate_of=dup_of))

    onesided: dict[tuple, list[dict]] = {}
    for r in gap_posts:
        onesided.setdefault((r["account"], r.get("player_key") or None, r.get("event_class")), []).append(r)
    for posts in onesided.values():
        stories.append(_one_sided(_earliest(posts), same_event_duplicate=False, duplicate_of=None))

    stories.sort(key=lambda s: s["story_id"])
    write_jsonl(data_dir / "stories.jsonl", stories)
    return stories


if __name__ == "__main__":
    import argparse

    from .config import sport_data_dir

    ap = argparse.ArgumentParser(description="Matching pipeline (candidates / adjudicate / resolve).")
    ap.add_argument("--resolve", action="store_true",
                    help="Run Stage 2+3 (LLM adjudication -> stories.jsonl). Requires ANTHROPIC_API_KEY.")
    ap.add_argument("--sport", default="mlb", help="Sport key from config (default: mlb).")
    args = ap.parse_args()
    data_dir = sport_data_dir(args.sport)

    cands = generate_candidates(data_dir=data_dir, sport=args.sport)
    print(f"Generated {len(cands)} candidate pair(s) -> {data_dir}/candidates.jsonl")
    for c in cands:
        lead = "RotoWire" if c["rotowire_first"] else "Underdog"
        print(f"  {c['player']} ({c['rotowire']['event_class']}/{c['underdog']['event_class']}): "
              f"{lead} first by {abs(c['time_delta_seconds'])}s")

    if args.resolve:
        stories = resolve_stories(data_dir=data_dir, sport=args.sport)
        matched = sum(1 for s in stories if s["status"] == "matched")
        print(f"Resolved {len(stories)} story(ies) ({matched} matched) -> {data_dir}/stories.jsonl")
