from __future__ import annotations

import json

from src import match
from src.anthropic_client import parse_verdict
from src.store import load_jsonl, write_jsonl


# ----------------------------- parse_verdict ------------------------------- #

def test_parse_verdict_plain_json():
    v = parse_verdict('{"same_story": true, "confidence": 0.92, "canonical_label": "Yelich to IL"}')
    assert v == {"same_story": True, "confidence": 0.92, "canonical_label": "Yelich to IL"}


def test_parse_verdict_strips_code_fences():
    raw = "```json\n{\"same_story\": false, \"confidence\": 0.1, \"canonical_label\": null}\n```"
    v = parse_verdict(raw)
    assert v["same_story"] is False and v["confidence"] == 0.1


def test_parse_verdict_recovers_embedded_object():
    v = parse_verdict('Sure! {"same_story": true, "confidence": 0.8, "canonical_label": "x"} hope that helps')
    assert v["same_story"] is True and v["confidence"] == 0.8


def test_parse_verdict_garbage_is_safe():
    v = parse_verdict("the model rambled with no json")
    assert v["same_story"] is False and v["confidence"] == 0.0 and v["parse_error"] is True


def test_parse_verdict_clamps_confidence():
    assert parse_verdict('{"same_story": true, "confidence": 5, "canonical_label": "x"}')["confidence"] == 1.0


# ------------------------- fakes / fixtures -------------------------------- #

class FakeAdjudicator:
    """Returns scripted verdicts and counts calls (to prove caching)."""

    def __init__(self, verdict):
        self.verdict_value = verdict
        self.calls = 0

    def verdict(self, player, rw_text, ud_text):
        self.calls += 1
        return dict(self.verdict_value)


PLAYER = "Christian Yelich"
PLAYER_KEY = "christian yelich"


def _news(tid, account, minute, player=PLAYER, event="injury_il", text="news"):
    from src.store import normalize_name
    return {
        "id": tid,
        "account": account,
        "created_at": f"2026-05-29T18:{minute:02d}:00.000Z",
        "text": text,
        "is_news": True,
        "event_class": event,
        "player": player,
        "team": "Brewers",
        "player_key": normalize_name(player),
        "players": [{"name": player, "team": "Brewers", "player_key": normalize_name(player)}],
        "public_metrics": {"impression_count": 100},
    }


def _candidate(rwid, udid, player_key, rw_min, ud_min):
    return {
        "candidate_id": f"{rwid}_{udid}_{player_key.replace(' ', '-')}",
        "player_key": player_key,
        "player": PLAYER,
        "team": "Brewers",
        "rotowire": {"tweet_id": rwid, "created_at": f"2026-05-29T18:{rw_min:02d}:00.000Z",
                     "event_class": "injury_il", "impression_count": 100},
        "underdog": {"tweet_id": udid, "created_at": f"2026-05-29T18:{ud_min:02d}:00.000Z",
                     "event_class": "injury_il", "impression_count": 50},
        "time_delta_seconds": (ud_min - rw_min) * 60,
        "rotowire_first": ud_min > rw_min,
        "event_compatible": True,
        "generated_at": "2026-05-29T19:00:00Z",
    }


def _ambiguous_tweets():
    # Same player, two RotoWire posts + one Underdog post in the window -> the
    # RW<->UD pairing is ambiguous, so text adjudication is required.
    return [
        _news("rw1", "RotoWireMLB", 0, PLAYER),
        _news("rw2", "RotoWireMLB", 5, PLAYER),
        _news("ud1", "UnderdogMLB", 8, PLAYER),
    ]


def _ambiguous_candidates():
    return [
        _candidate("rw1", "ud1", PLAYER_KEY, 0, 8),
        _candidate("rw2", "ud1", PLAYER_KEY, 5, 8),
    ]


# --------------------------- adjudicate ------------------------------------ #

def test_adjudicate_judges_clean_pair_and_caches(tmp_path):
    # Even a clean 1:1 is sent to the LLM (to confirm the player across spelling /
    # nickname variants) and the verdict is cached.
    write_jsonl(tmp_path / "tweets.jsonl",
                [_news("rw1", "RotoWireMLB", 0, PLAYER, text="Christian Yelich placed on the IL"),
                 _news("ud1", "UnderdogMLB", 8, PLAYER)])
    write_jsonl(tmp_path / "candidates.jsonl", [_candidate("rw1", "ud1", PLAYER_KEY, 0, 8)])
    fake = FakeAdjudicator({"same_story": True, "confidence": 0.9, "canonical_label": "Yelich IL"})

    judged = match.adjudicate_candidates(tmp_path, adjudicator=fake, min_confidence=0.75)
    assert judged[0]["accepted"] is True
    assert judged[0]["match_confidence"] == 0.9
    assert judged[0]["match_source"] == "llm"
    assert judged[0]["canonical_label"] == "Yelich IL"
    assert fake.calls == 1

    # Re-run: verdict served from cache, no extra LLM call.
    match.adjudicate_candidates(tmp_path, adjudicator=fake, min_confidence=0.75)
    assert fake.calls == 1
    assert len(load_jsonl(tmp_path / "verdicts.jsonl")) == 1


def test_adjudicate_can_reject_clean_pair_when_not_same_story(tmp_path):
    # The LLM is the authority: a 1:1 it deems NOT the same story is rejected.
    write_jsonl(tmp_path / "tweets.jsonl",
                [_news("rw1", "RotoWireMLB", 0, PLAYER), _news("ud1", "UnderdogMLB", 8, PLAYER)])
    write_jsonl(tmp_path / "candidates.jsonl", [_candidate("rw1", "ud1", PLAYER_KEY, 0, 8)])
    fake = FakeAdjudicator({"same_story": False, "confidence": 0.1, "canonical_label": None})

    judged = match.adjudicate_candidates(tmp_path, adjudicator=fake, min_confidence=0.75)
    assert judged[0]["accepted"] is False
    assert fake.calls == 1


def test_adjudicate_judges_every_pair_when_player_has_multiple_posts(tmp_path):
    write_jsonl(tmp_path / "tweets.jsonl", _ambiguous_tweets())
    write_jsonl(tmp_path / "candidates.jsonl", _ambiguous_candidates())
    fake = FakeAdjudicator({"same_story": True, "confidence": 0.9, "canonical_label": "Yelich IL"})

    judged = match.adjudicate_candidates(tmp_path, adjudicator=fake, min_confidence=0.75)
    assert all(j["accepted"] for j in judged)
    assert all(j["match_source"] == "llm" for j in judged)
    assert fake.calls == 2  # one verdict per pair

    # Re-run: verdicts served from cache, no extra LLM calls.
    match.adjudicate_candidates(tmp_path, adjudicator=fake, min_confidence=0.75)
    assert fake.calls == 2
    assert len(load_jsonl(tmp_path / "verdicts.jsonl")) == 2


def test_adjudicate_rejects_below_threshold(tmp_path):
    write_jsonl(tmp_path / "tweets.jsonl", _ambiguous_tweets())
    write_jsonl(tmp_path / "candidates.jsonl", _ambiguous_candidates())
    fake = FakeAdjudicator({"same_story": True, "confidence": 0.5, "canonical_label": "maybe"})
    judged = match.adjudicate_candidates(tmp_path, adjudicator=fake, min_confidence=0.75)
    assert all(j["accepted"] is False for j in judged)


# ----------------------------- resolve ------------------------------------- #

def test_resolve_builds_matched_story(tmp_path):
    # Clean 1:1 -> judged by the adjudicator -> matched story (match_method "llm").
    write_jsonl(tmp_path / "tweets.jsonl",
                [_news("rw1", "RotoWireMLB", 0, PLAYER, text="Yelich to IL"),
                 _news("ud1", "UnderdogMLB", 8, PLAYER)])
    write_jsonl(tmp_path / "candidates.jsonl", [_candidate("rw1", "ud1", PLAYER_KEY, 0, 8)])
    fake = FakeAdjudicator({"same_story": True, "confidence": 0.9, "canonical_label": "Yelich IL"})

    stories = match.resolve_stories(tmp_path, adjudicator=fake)
    assert len(stories) == 1
    s = stories[0]
    assert s["status"] == "matched" and s["match_method"] == "llm"
    assert s["time_delta_seconds"] == 480 and s["rotowire_first"] is True
    assert s["rotowire"]["tweet_id"] == "rw1" and s["underdog"]["tweet_id"] == "ud1"
    assert s["match_confidence"] == 0.9 and s["canonical_label"] == "Yelich IL"


def test_resolve_one_to_one_closest_pair_wins(tmp_path):
    # RotoWire posted twice (rw1 @0, rw2 @5); both pair with ud1 @8. Strict 1-to-1:
    # the CLOSEST pair (rw2<->ud1, 180s) wins; rw1 becomes a same-event duplicate, NOT
    # a second match and NOT a true coverage gap.
    write_jsonl(tmp_path / "tweets.jsonl", _ambiguous_tweets())
    write_jsonl(tmp_path / "candidates.jsonl", _ambiguous_candidates())
    fake = FakeAdjudicator({"same_story": True, "confidence": 0.9, "canonical_label": "Yelich IL"})

    stories = match.resolve_stories(tmp_path, adjudicator=fake)
    matched = [s for s in stories if s["status"] == "matched"]
    assert len(matched) == 1
    m = matched[0]
    assert m["rotowire"]["tweet_id"] == "rw2"  # closest RW post, not the earliest
    assert m["underdog"]["tweet_id"] == "ud1"
    assert m["time_delta_seconds"] == 180
    assert m["match_method"] == "llm"

    dups = [s for s in stories if s.get("same_event_duplicate")]
    assert len(dups) == 1
    d = dups[0]
    assert d["status"] == "rotowire_only" and d["rotowire"]["tweet_id"] == "rw1"
    assert d["duplicate_of"] == m["story_id"]


def test_resolve_rejected_pairs_become_one_sided(tmp_path):
    write_jsonl(tmp_path / "tweets.jsonl", _ambiguous_tweets())
    write_jsonl(tmp_path / "candidates.jsonl", _ambiguous_candidates())
    fake = FakeAdjudicator({"same_story": False, "confidence": 0.2, "canonical_label": None})

    stories = match.resolve_stories(tmp_path, adjudicator=fake)
    statuses = sorted(s["status"] for s in stories)
    assert statuses == ["rotowire_only", "underdog_only"]
    rw_only = next(s for s in stories if s["status"] == "rotowire_only")
    assert rw_only["underdog"] is None and rw_only["rotowire"]["tweet_id"] == "rw1"


def test_resolve_unmatched_news_is_coverage_gap(tmp_path):
    # No candidates at all; a lone RotoWire news post -> rotowire_only.
    write_jsonl(tmp_path / "tweets.jsonl", [_news("rw1", "RotoWireMLB", 0, PLAYER)])
    write_jsonl(tmp_path / "candidates.jsonl", [])
    stories = match.resolve_stories(tmp_path, adjudicator=FakeAdjudicator({}))
    assert len(stories) == 1 and stories[0]["status"] == "rotowire_only"


def test_exact_name_standin_accepts_without_caching(tmp_path):
    # No-key stand-in: accepts by shared player + window, never writes the cache.
    write_jsonl(tmp_path / "tweets.jsonl",
                [_news("rw1", "RotoWireMLB", 0, PLAYER, text="Yelich to the IL"),
                 _news("ud1", "UnderdogMLB", 8, PLAYER)])
    write_jsonl(tmp_path / "candidates.jsonl", [_candidate("rw1", "ud1", PLAYER_KEY, 0, 8)])

    stories = match.resolve_stories(
        tmp_path, adjudicator=match.ExactNameAdjudicator(),
        method="exact_name", cache=False,
    )
    assert len(stories) == 1
    s = stories[0]
    assert s["status"] == "matched" and s["match_method"] == "exact_name"
    assert s["match_confidence"] == 1.0
    # Stand-in must NOT write the verdict cache (real LLM still runs later).
    assert not (tmp_path / "verdicts.jsonl").exists()


def test_resolve_story_ids_are_stable_across_runs(tmp_path):
    write_jsonl(tmp_path / "tweets.jsonl",
                [_news("rw1", "RotoWireMLB", 0, PLAYER), _news("ud1", "UnderdogMLB", 8, PLAYER)])
    write_jsonl(tmp_path / "candidates.jsonl", [_candidate("rw1", "ud1", PLAYER_KEY, 0, 8)])
    fake = FakeAdjudicator({"same_story": True, "confidence": 0.9, "canonical_label": "Yelich IL"})

    first = match.resolve_stories(tmp_path, adjudicator=fake)[0]["story_id"]
    second = match.resolve_stories(tmp_path, adjudicator=fake)[0]["story_id"]
    assert first == second == "st_rw1_ud1"
