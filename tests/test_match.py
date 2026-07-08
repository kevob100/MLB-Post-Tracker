from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import match as match_mod
from src.store import load_jsonl, write_jsonl

CFG = {
    "sports": {
        "mlb": {
            "label": "MLB",
            "active": True,
            "accounts": {
                "rotowire": {"handle": "RotoWireMLB", "user_id": "111"},
                "underdog": {"handle": "UnderdogMLB", "user_id": "222"},
            },
        },
    },
    "matching": {
        "time_window_minutes": 30,
        "min_match_confidence": 0.75,
        "compatible_event_classes": {
            "injury": ["injury_il", "injury_dtd", "scratch", "return"],
            "transaction": [
                "transaction_callup", "transaction_option", "transaction_dfa",
                "transaction_trade", "transaction_sign_release",
            ],
            "role": ["role_change"],
        },
    },
}


@pytest.fixture(autouse=True)
def _patch_config(monkeypatch):
    monkeypatch.setattr(match_mod, "load_config", lambda: CFG)


def _ts(base_min: float) -> str:
    dt = datetime(2026, 5, 29, 18, 0, tzinfo=timezone.utc) + timedelta(minutes=base_min)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _post(tid: str, account: str, minute: float, player: str, event: str) -> dict:
    from src.store import normalize_name
    return {
        "id": tid,
        "account": account,
        "created_at": _ts(minute),
        "is_news": True,
        "event_class": event,
        "public_metrics": {"impression_count": 1000},
        "player": player,
        "team": "Brewers",
        "player_key": normalize_name(player),
        "players": [{"name": player, "team": "Brewers", "player_key": normalize_name(player)}],
    }


def _seed(tmp_path, posts):
    write_jsonl(tmp_path / "tweets.jsonl", posts)


def test_true_match_generates_candidate_with_correct_delta(tmp_path):
    # RotoWire at +0, Underdog at +8 -> RotoWire first, delta = +480s.
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 0, "Christian Yelich", "injury_il"),
        _post("2", "UnderdogMLB", 8, "Christian Yelich", "injury_il"),
    ])
    cands = match_mod.generate_candidates(data_dir=tmp_path)
    assert len(cands) == 1
    c = cands[0]
    assert c["player_key"] == "christian yelich"
    assert c["time_delta_seconds"] == 480
    assert c["rotowire_first"] is True


def test_underdog_first_negative_delta(tmp_path):
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 10, "Christian Yelich", "injury_il"),
        _post("2", "UnderdogMLB", 2, "Christian Yelich", "injury_il"),
    ])
    c = match_mod.generate_candidates(data_dir=tmp_path)[0]
    assert c["time_delta_seconds"] == -480
    assert c["rotowire_first"] is False


def test_compatible_event_group_matches(tmp_path):
    # injury_il vs injury_dtd are in the same group -> still a candidate.
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 0, "Christian Yelich", "injury_il"),
        _post("2", "UnderdogMLB", 5, "Christian Yelich", "injury_dtd"),
    ])
    assert len(match_mod.generate_candidates(data_dir=tmp_path)) == 1


def test_incompatible_event_classes_still_candidate_but_flagged(tmp_path):
    # Event class no longer gates candidacy: same player within the window is a
    # candidate regardless, but event_compatible records that the classes differ.
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 0, "Christian Yelich", "transaction_trade"),
        _post("2", "UnderdogMLB", 5, "Christian Yelich", "injury_il"),
    ])
    cands = match_mod.generate_candidates(data_dir=tmp_path)
    assert len(cands) == 1
    assert cands[0]["event_compatible"] is False


def test_outside_time_window_rejected(tmp_path):
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 0, "Christian Yelich", "injury_il"),
        _post("2", "UnderdogMLB", 31, "Christian Yelich", "injury_il"),
    ])
    assert match_mod.generate_candidates(data_dir=tmp_path) == []


def test_different_players_rejected(tmp_path):
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 0, "Christian Yelich", "injury_il"),
        _post("2", "UnderdogMLB", 5, "Willy Adames", "injury_il"),
    ])
    assert match_mod.generate_candidates(data_dir=tmp_path) == []


def test_non_news_posts_ignored(tmp_path):
    rw = _post("1", "RotoWireMLB", 0, "Christian Yelich", "injury_il")
    ud = _post("2", "UnderdogMLB", 5, "Christian Yelich", "injury_il")
    ud["is_news"] = False  # excluded post must not pair
    _seed(tmp_path, [rw, ud])
    assert match_mod.generate_candidates(data_dir=tmp_path) == []


def test_idempotent_rerun(tmp_path):
    _seed(tmp_path, [
        _post("1", "RotoWireMLB", 0, "Christian Yelich", "injury_il"),
        _post("2", "UnderdogMLB", 8, "Christian Yelich", "injury_il"),
    ])
    first = match_mod.generate_candidates(data_dir=tmp_path)
    stamp = first[0]["generated_at"]
    second = match_mod.generate_candidates(data_dir=tmp_path)
    rows = load_jsonl(tmp_path / "candidates.jsonl")
    assert len(rows) == 1
    assert second[0]["generated_at"] == stamp  # discovery time preserved
