from __future__ import annotations

from src import classify
from src.classify import parse_classification
from src.store import load_jsonl, write_jsonl


# --------------------------- parse_classification -------------------------- #

def test_parse_plain_json_news():
    c = parse_classification(
        '{"is_news": true, "player": "Christian Yelich", "team": "Brewers", '
        '"event_class": "injury_il", "excluded_reason": null}'
    )
    assert c["is_news"] is True
    assert c["player"] == "Christian Yelich"
    assert c["event_class"] == "injury_il"
    assert c["excluded_reason"] is None


def test_parse_strips_code_fences():
    raw = ("```json\n{\"is_news\": false, \"player\": null, \"team\": null, "
           "\"event_class\": \"other\", \"excluded_reason\": \"promo\"}\n```")
    c = parse_classification(raw)
    assert c["is_news"] is False and c["excluded_reason"] == "promo"


def test_parse_recovers_embedded_object():
    c = parse_classification(
        'Sure: {"is_news": true, "player": "Eury Perez", "team": "Marlins", '
        '"event_class": "return", "excluded_reason": null} done'
    )
    assert c["is_news"] is True and c["event_class"] == "return"


def test_parse_garbage_is_safe():
    c = parse_classification("model rambled")
    assert c["is_news"] is False and c["parse_error"] is True


def test_parse_unknown_event_class_falls_back_to_other_and_not_news():
    c = parse_classification(
        '{"is_news": true, "player": "X", "event_class": "made_up", "excluded_reason": null}'
    )
    # An out-of-vocab class collapses to "other", which cannot be news.
    assert c["event_class"] == "other" and c["is_news"] is False


def test_parse_news_requires_a_player():
    c = parse_classification(
        '{"is_news": true, "player": null, "event_class": "injury_il", "excluded_reason": null}'
    )
    assert c["is_news"] is False and c["excluded_reason"] == "no_player"


# ------------------------------ classify_file ------------------------------ #

class FakeClassifier:
    """Returns scripted verdicts keyed by tweet id; counts calls (to prove caching)."""

    def __init__(self, by_id):
        self.by_id = by_id
        self.calls = 0
        self._next = None

    def classify(self, text):
        self.calls += 1
        # The classifier sees only text, so we map text -> verdict here.
        return dict(self._lookup(text))

    def _lookup(self, text):
        return self.by_id[text]


def _post(tid, text):
    return {"id": tid, "account": "RotoWireMLB", "created_at": f"2026-05-29T18:0{tid}:00.000Z",
            "text": text, "public_metrics": {"impression_count": 1}}


def test_classify_file_enriches_and_caches(tmp_path):
    write_jsonl(tmp_path / "tweets.jsonl", [
        _post("1", "Yelich placed on the IL"),
        _post("2", "Use code WIN for a free pick"),
    ])
    fake = FakeClassifier({
        "Yelich placed on the IL": {"is_news": True, "player": "Christian Yelich",
                                    "team": "Brewers", "event_class": "injury_il",
                                    "excluded_reason": None},
        "Use code WIN for a free pick": {"is_news": False, "player": None, "team": None,
                                         "event_class": "other", "excluded_reason": "promo"},
    })

    summary = classify.classify_file(tmp_path, classifier=fake)
    assert summary == {"total": 2, "news": 1, "excluded": {"promo": 1}}
    assert fake.calls == 2

    rows = {r["id"]: r for r in load_jsonl(tmp_path / "tweets.jsonl")}
    assert rows["1"]["is_news"] is True
    assert rows["1"]["player"] == "Christian Yelich"
    assert rows["1"]["player_key"] == "christian yelich"
    assert rows["1"]["players"][0]["player_key"] == "christian yelich"
    assert rows["2"]["is_news"] is False and rows["2"]["excluded_reason"] == "promo"

    # Re-run: verdicts served from data/classifications.jsonl, no new LLM calls.
    classify.classify_file(tmp_path, classifier=fake)
    assert fake.calls == 2
    assert len(load_jsonl(tmp_path / "classifications.jsonl")) == 2


def test_classify_file_no_key_uses_safe_default(tmp_path):
    # llm=False must never call the classifier; uncached posts become non-news.
    write_jsonl(tmp_path / "tweets.jsonl", [_post("1", "anything")])
    summary = classify.classify_file(tmp_path, llm=False)
    assert summary["news"] == 0
    row = load_jsonl(tmp_path / "tweets.jsonl")[0]
    assert row["is_news"] is False and row["player"] is None
    # Nothing cached, since no verdict was produced.
    assert not (tmp_path / "classifications.jsonl").exists()
