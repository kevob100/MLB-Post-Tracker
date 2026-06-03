from __future__ import annotations

import json

import pytest

from src import review
from src.aggregate import apply_reviews
from src.store import load_jsonl, write_jsonl


def _matched(sid, conf=0.8, day=29):
    rw = f"2026-05-{day:02d}T18:00:00.000Z"
    ud = f"2026-05-{day:02d}T18:05:00.000Z"
    return {
        "story_id": sid,
        "status": "matched",
        "player": "Christian Yelich",
        "team": "Brewers",
        "event_class": "injury_dtd",
        "canonical_label": f"story {sid}",
        "match_confidence": conf,
        "rotowire": {"tweet_id": f"rw{sid}", "created_at": rw, "impression_count": 100},
        "underdog": {"tweet_id": f"ud{sid}", "created_at": ud, "impression_count": 50},
        "time_delta_seconds": 300,
        "rotowire_first": True,
    }


def _one_sided(sid):
    return {
        "story_id": sid,
        "status": "rotowire_only",
        "player": "Someone",
        "match_confidence": None,
        "rotowire": {"tweet_id": f"x{sid}", "created_at": "2026-05-29T18:00:00.000Z", "impression_count": 10},
        "underdog": None,
        "time_delta_seconds": None,
        "rotowire_first": None,
    }


def test_make_review_shapes_and_validates():
    r = review.make_review("a", "confirm", note="ok")
    assert r["decision"] == "confirm" and r["merge_into"] is None and r["note"] == "ok"
    assert r["reviewed_at"]  # timestamped

    m = review.make_review("b", "merge", merge_into="a")
    assert m["merge_into"] == "a"

    with pytest.raises(ValueError):
        review.make_review("a", "bogus")
    with pytest.raises(ValueError):
        review.make_review("a", "merge")  # missing target


def test_pending_only_matched_pending_sorted_worst_first():
    stories = apply_reviews([_matched("a", 0.95), _matched("b", 0.6), _one_sided("c")], [])
    q = review.pending(stories)
    assert [s["story_id"] for s in q] == ["b", "a"]  # one-sided excluded; low conf first


def test_pending_excludes_already_reviewed():
    reviews = [{"story_id": "a", "decision": "confirm", "reviewed_at": "2026-05-29T20:00:00Z"}]
    stories = apply_reviews([_matched("a"), _matched("b")], reviews)
    assert [s["story_id"] for s in review.pending(stories)] == ["b"]


def test_pending_min_confidence_filter():
    stories = apply_reviews([_matched("a", 0.95), _matched("b", 0.6)], [])
    assert [s["story_id"] for s in review.pending(stories, min_confidence=0.9)] == ["b"]


def test_run_review_appends_and_rebuilds(tmp_path):
    data_dir = tmp_path
    out_dir = tmp_path / "docs_data"
    write_jsonl(data_dir / "stories.jsonl", [_matched("a", 0.6), _matched("b", 0.9)])
    write_jsonl(data_dir / "reviews.jsonl", [])

    # 'a' (worst conf, seen first) rejected; 'b' confirmed.
    scripted = iter([{"decision": "reject"}, {"decision": "confirm"}])
    written = review.run_review(
        data_dir=data_dir, docs_data_dir=out_dir,
        prompt_fn=lambda s, ids: next(scripted), out=lambda *_: None,
    )
    assert [w["decision"] for w in written] == ["reject", "confirm"]

    reviews = load_jsonl(data_dir / "reviews.jsonl")
    assert {r["story_id"]: r["decision"] for r in reviews} == {"a": "reject", "b": "confirm"}

    agg = json.loads((out_dir / "aggregates.json").read_text())
    assert agg["summary"]["matched"] == 1  # 'a' rejected, 'b' confirmed kept


def test_run_review_stop_leaves_rest_pending(tmp_path):
    data_dir = tmp_path
    write_jsonl(data_dir / "stories.jsonl", [_matched("a", 0.6), _matched("b", 0.9)])

    written = review.run_review(
        data_dir=data_dir, docs_data_dir=tmp_path / "docs_data",
        prompt_fn=lambda s, ids: review.STOP, rebuild=False, out=lambda *_: None,
    )
    assert written == []
    assert load_jsonl(data_dir / "reviews.jsonl") == []


def test_run_review_skip_writes_nothing(tmp_path):
    data_dir = tmp_path
    write_jsonl(data_dir / "stories.jsonl", [_matched("a")])
    written = review.run_review(
        data_dir=data_dir, docs_data_dir=tmp_path / "d",
        prompt_fn=lambda s, ids: None, rebuild=False, out=lambda *_: None,
    )
    assert written == [] and load_jsonl(data_dir / "reviews.jsonl") == []


def test_run_review_no_pending_noop(tmp_path):
    write_jsonl(tmp_path / "stories.jsonl", [_one_sided("c")])
    written = review.run_review(
        data_dir=tmp_path, docs_data_dir=tmp_path / "d",
        prompt_fn=lambda s, ids: pytest.fail("should not prompt"), out=lambda *_: None,
    )
    assert written == []


def test_run_review_merge_records_target(tmp_path):
    data_dir = tmp_path
    write_jsonl(data_dir / "stories.jsonl", [_matched("a"), _matched("b")])
    scripted = iter([
        {"decision": "merge", "merge_into": "b"},
        {"decision": "confirm"},
    ])
    review.run_review(
        data_dir=data_dir, docs_data_dir=tmp_path / "d",
        prompt_fn=lambda s, ids: next(scripted), rebuild=False, out=lambda *_: None,
    )
    reviews = {r["story_id"]: r for r in load_jsonl(data_dir / "reviews.jsonl")}
    assert reviews["a"]["decision"] == "merge" and reviews["a"]["merge_into"] == "b"
