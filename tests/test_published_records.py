from __future__ import annotations

import json
import sys

import run


def record(scenario: str, engine: str, n: int, replicate: int, **extra) -> dict:
    return {
        "status": "ok", "engine": engine, "n": n, "days": 100, "replicate": replicate,
        "fingerprint": f"{scenario}-{engine}-{n}-{replicate}",
        "format_version": run.RUNNER_FORMAT_VERSION, "simulate_seconds": 0.5,
    } | extra


def publish(root, scenario: str, engine: str, n: int, replicate: int, **extra) -> None:
    path = root / scenario / engine / f"n{n}" / f"replicate-{replicate:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record(scenario, engine, n, replicate, **extra)))


def test_full_records_are_published_and_smoke_records_are_not() -> None:
    assert run.record_path("full", "scenario_01", "ixa", 10000, 7) == (
        run.RESULTS_DIR / "runs" / "scenario_01" / "ixa" / "n10000" / "replicate-007.json"
    )
    assert run.record_path("smoke", "scenario_01", "ixa", 1000, 1) == (
        run.CACHE_DIR / "results" / "scenario_01" / "ixa" / "n1000" / "replicate-001.json"
    )


def test_results_are_derived_from_published_records_only(tmp_path, monkeypatch) -> None:
    runs, cache, results = tmp_path / "runs", tmp_path / "cache", tmp_path / "results"
    monkeypatch.setattr(run, "RUNS_DIR", runs)
    monkeypatch.setattr(run, "CACHE_DIR", cache)
    monkeypatch.setattr(run, "RESULTS_DIR", results)
    publish(runs, "scenario_00", "ixa", 10000, 2)
    publish(runs, "scenario_00", "ixa", 10000, 1)
    publish(runs, "scenario_02", "EoN", 10000, 1,
            daily_incidence=[1, 3], reproductive_number=[2.0, 1.0, None])
    # A smoke record in the cache is not published.
    publish(cache / "results", "scenario_00", "ixa", 1000, 1)
    assert run.collect_results() == 3
    first = (results / "results.csv").read_text()
    rows = first.splitlines()[1:]
    assert [row.split(",")[:6:5] for row in rows] == [
        ["scenario_00", "1"], ["scenario_00", "2"], ["scenario_02", "1"],
    ]
    daily = (results / "daily.csv").read_text().splitlines()
    assert daily[1:] == ["scenario_02,EoN,10000,0,,2.0", "scenario_02,EoN,10000,1,1,1.0",
                         "scenario_02,EoN,10000,2,3,"]
    # Collecting again on another clone with the same records gives the same files.
    run.collect_results()
    assert (results / "results.csv").read_text() == first


def test_a_clone_without_the_cache_skips_published_replicates(tmp_path, monkeypatch, capsys) -> None:
    """The committed records, not the local cache, decide what is pending."""
    runs = tmp_path / "runs"
    monkeypatch.setattr(run, "RUNS_DIR", runs)
    monkeypatch.setattr(run, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(run, "engine_versions", lambda engines, scenarios: {e: "1" for e in engines})
    monkeypatch.setattr(run, "collect_environment",
                        lambda versions, workers, profile: {"environment_id": "e"})
    arguments = ["run.py", "--profile", "full", "--scenarios", "scenario_00",
                 "--engines", "EoN", "--sizes", "10000", "--replicates", "2", "--dry-run"]
    monkeypatch.setattr(sys, "argv", arguments)
    assert run.main() == 0
    assert "cached=0; pending=2" in capsys.readouterr().out

    # Publish replicate 1 exactly as the planner fingerprints it.
    planned = {}
    real_fingerprint = run.fingerprint
    def remember(identity):
        value = real_fingerprint(identity)
        planned[identity["replicate"]] = value
        return value
    monkeypatch.setattr(run, "fingerprint", remember)
    assert run.main() == 0
    capsys.readouterr()
    publish(runs, "scenario_00", "EoN", 10000, 1, fingerprint=planned[1])
    assert run.main() == 0
    assert "cached=1; pending=1" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", arguments + ["--force"])
    assert run.main() == 0
    assert "cached=0; pending=2" in capsys.readouterr().out


def test_collecting_nothing_keeps_the_committed_results(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "results.csv").write_text("committed\n")
    (results / "daily.csv").write_text("committed\n")
    monkeypatch.setattr(run, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(run, "RESULTS_DIR", results)
    assert run.collect_results() == 0
    assert (results / "results.csv").read_text() == "committed\n"
    assert (results / "daily.csv").read_text() == "committed\n"
