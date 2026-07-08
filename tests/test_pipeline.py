from __future__ import annotations

import json

from src import pipeline


def _cfg(*sport_keys):
    return {
        "sports": {
            key: {
                "label": key.upper(),
                "active": True,
                "accounts": {
                    "rotowire": {"handle": f"RotoWire{key.upper()}", "user_id": "1"},
                    "underdog": {"handle": f"Underdog{key.upper()}", "user_id": "2"},
                },
            }
            for key in sport_keys
        }
    }


def _patch_stages(monkeypatch, calls, *, classified=None, resolved=None):
    """Patch every pipeline stage with a recorder that tolerates the new kwargs."""
    monkeypatch.setattr(pipeline.collect, "collect",
                        lambda **kw: (calls.append(("collect", kw.get("sport"))), {"added": 0})[1])

    def _classify(**kw):
        calls.append(("classify", kw.get("sport")))
        if classified is not None:
            classified.append(kw)
        return {"total": 0, "news": 0, "excluded": {}}
    monkeypatch.setattr(pipeline.classify, "classify_file", _classify)

    monkeypatch.setattr(pipeline.match, "generate_candidates",
                        lambda **kw: (calls.append(("match", kw.get("sport"))), [])[1])

    def _resolve(**kw):
        calls.append(("resolve", kw.get("sport")))
        if resolved is not None:
            resolved.append(kw)
        return []
    monkeypatch.setattr(pipeline.match, "resolve_stories", _resolve)

    sentinel = {"generated_at": "2026-07-08T00:00:00+00:00",
                "summary": {"matched": 0, "rotowire_first_rate": None,
                            "median_lead_seconds": None, "rotowire_only": 0, "underdog_only": 0}}
    monkeypatch.setattr(pipeline.aggregate, "build_aggregates",
                        lambda **kw: (calls.append(("aggregate", None)), sentinel)[1])


def _patch_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "sport_data_dir", lambda s: tmp_path / "data" / s)
    monkeypatch.setattr(pipeline, "sport_docs_dir", lambda s: tmp_path / "docs" / s)
    monkeypatch.setattr(pipeline, "DOCS_DATA_DIR", tmp_path / "docs")


def test_run_invokes_stages_in_order(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # no key -> stand-in path
    monkeypatch.setattr(pipeline, "load_config", lambda: _cfg("mlb"))
    _patch_paths(monkeypatch, tmp_path)
    calls = []
    _patch_stages(monkeypatch, calls)

    result = pipeline.run()

    assert [c[0] for c in calls] == ["collect", "classify", "match", "resolve", "aggregate"]
    assert set(result) == {"mlb"}
    out = capsys.readouterr().out
    assert "[collect]" in out and "[classify]" in out and "[aggregate]" in out
    assert "exact_name" in out  # no-key path uses the exact-name stand-in

    # A sports.json index is written for the dashboard switcher.
    index = json.loads((tmp_path / "docs" / "sports.json").read_text())
    assert [s["key"] for s in index["sports"]] == ["mlb"]
    assert index["sports"][0]["handles"]["rotowire"] == "RotoWireMLB"


def test_run_loops_every_active_sport(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "load_config", lambda: _cfg("mlb", "nfl", "nba"))
    _patch_paths(monkeypatch, tmp_path)
    calls = []
    _patch_stages(monkeypatch, calls)

    result = pipeline.run()

    assert set(result) == {"mlb", "nfl", "nba"}
    # Each sport runs the full stage sequence; collect fires once per sport.
    assert [s for (stage, s) in calls if stage == "collect"] == ["mlb", "nfl", "nba"]
    index = json.loads((tmp_path / "docs" / "sports.json").read_text())
    assert [s["key"] for s in index["sports"]] == ["mlb", "nfl", "nba"]


def test_run_single_sport_keeps_full_index(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "load_config", lambda: _cfg("mlb", "nfl"))
    _patch_paths(monkeypatch, tmp_path)
    calls = []
    _patch_stages(monkeypatch, calls)

    pipeline.run(sport="nfl")

    # Only NFL runs...
    assert [s for (stage, s) in calls if stage == "collect"] == ["nfl"]
    # ...but the index still lists all active sports so the switcher stays complete.
    index = json.loads((tmp_path / "docs" / "sports.json").read_text())
    assert [s["key"] for s in index["sports"]] == ["mlb", "nfl"]


def test_run_resolves_with_llm_when_key_present(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline, "load_config", lambda: _cfg("mlb"))
    _patch_paths(monkeypatch, tmp_path)
    calls, classified, resolved = [], [], []
    _patch_stages(monkeypatch, calls, classified=classified, resolved=resolved)

    pipeline.run()

    assert len(resolved) == 1
    # With the key, classification runs against the LLM (not the cached-only fallback).
    assert classified and classified[0].get("llm", True) is not False
