from __future__ import annotations

from src import pipeline


def test_run_invokes_stages_in_order(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)  # no key -> stand-in path
    calls = []

    monkeypatch.setattr(pipeline.collect, "collect",
                        lambda data_dir: (calls.append("collect"), {"added": 0})[1])
    monkeypatch.setattr(pipeline.classify, "classify_file",
                        lambda data_dir, **kw: (calls.append("classify"),
                                                {"total": 0, "news": 0, "excluded": {}})[1])
    monkeypatch.setattr(pipeline.match, "generate_candidates",
                        lambda data_dir: (calls.append("match"), [])[1])
    monkeypatch.setattr(pipeline.match, "resolve_stories",
                        lambda data_dir, **kw: (calls.append("resolve"), [])[1])
    sentinel = {"summary": {"matched": 0, "rotowire_first_rate": None,
                            "median_lead_seconds": None, "rotowire_only": 0, "underdog_only": 0}}
    monkeypatch.setattr(pipeline.aggregate, "build_aggregates",
                        lambda data_dir: (calls.append("aggregate"), sentinel)[1])

    result = pipeline.run(data_dir=tmp_path)

    assert calls == ["collect", "classify", "match", "resolve", "aggregate"]
    assert result is sentinel
    out = capsys.readouterr().out
    assert "[collect]" in out and "[classify]" in out and "[aggregate]" in out
    assert "exact_name" in out  # no-key path uses the exact-name stand-in


def test_run_resolves_with_llm_when_key_present(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(pipeline.collect, "collect", lambda data_dir: {})
    classified = []
    monkeypatch.setattr(pipeline.classify, "classify_file",
                        lambda data_dir, **kw: (classified.append(kw),
                                                {"total": 0, "news": 0, "excluded": {}})[1])
    monkeypatch.setattr(pipeline.match, "generate_candidates", lambda data_dir: [])
    resolved = []
    monkeypatch.setattr(pipeline.match, "resolve_stories",
                        lambda data_dir, **kw: (resolved.append(True), [])[1])
    monkeypatch.setattr(pipeline.aggregate, "build_aggregates",
                        lambda data_dir: {"summary": {"matched": 0, "rotowire_first_rate": None,
                                                       "median_lead_seconds": None,
                                                       "rotowire_only": 0, "underdog_only": 0}})
    pipeline.run(data_dir=tmp_path)
    assert resolved == [True]
    # With the key, classification runs against the LLM (not the cached-only fallback).
    assert classified and classified[0].get("llm", True) is not False
