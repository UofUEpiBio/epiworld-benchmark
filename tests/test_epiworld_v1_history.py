from __future__ import annotations

import csv
import json

import pytest

from scripts import epiworld_v1_history as history


def write_records(runs, engine, simulate, environment="e" * 64, replicates=3, version=None):
    for replicate in range(1, replicates + 1):
        path = runs / "scenario_00" / engine / "n100" / f"replicate-{replicate:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": "ok", "engine_version": version or f"{engine}-version", "n": 100,
            "simulate_seconds": simulate * replicate, "setup_seconds": 0.5, "read_seconds": 0.25,
            "peak_rss_setup_bytes": 2**20, "peak_rss_simulate_bytes": 3 * 2**20,
            "process_peak_rss_bytes": None, "final_susceptible": 60,
            "environment_id": environment,
        }))


def test_a_stage_adds_the_medians_of_both_engines(tmp_path, monkeypatch, capsys) -> None:
    runs, doc = tmp_path / "runs", tmp_path / "epiworld-v1.md"
    monkeypatch.setattr(history, "RUNS_DIR", runs)
    monkeypatch.setattr(history, "HISTORY", tmp_path / "history.csv")
    monkeypatch.setattr(history, "DOC", doc)
    monkeypatch.setattr(history, "ROOT", tmp_path)
    monkeypatch.setattr(history, "ENVIRONMENTS", tmp_path / "environments")
    monkeypatch.setattr(history, "STAGE_ENVIRONMENTS", tmp_path / "stages")
    (tmp_path / "environments").mkdir()
    (tmp_path / "environments" / "scenario_00.json").write_text(
        json.dumps({"environment_id": "e" * 64, "executed": 6})
    )
    doc.write_text(f"# Title\n\n{history.BEGIN}\nold\n{history.END}\n\n## Notes\n")
    write_records(runs, "epiworld", 1.0)
    write_records(runs, "epiworld-v1", 1.5)

    def stage(name):
        monkeypatch.setattr("sys.argv", ["epiworld_v1_history.py", name])
        assert history.main() == 0
        with (tmp_path / "history.csv").open() as stream:
            return list(csv.DictReader(stream))

    rows = stage("catch-up")
    assert [(row["stage"], row["scenario"], row["n"], row["replicates"]) for row in rows] == [
        ("catch-up", "scenario_00", "100", "3")
    ]
    row = rows[0]
    assert float(row["simulate_seconds_0x"]) == 2.0
    assert float(row["simulate_seconds_v1"]) == 3.0
    assert float(row["simulate_ratio"]) == 1.5
    assert float(row["build_seconds_v1"]) == 0.25
    assert float(row["peak_mib_v1"]) == 3.0
    assert float(row["attack_rate_0x"]) == 0.4
    saved = json.loads((tmp_path / "stages" / "catch-up.json").read_text())
    assert saved == {"scenario_00": {"environment_id": "e" * 64, "executed": 6}}
    text = doc.read_text()
    assert "old" not in text and text.endswith("## Notes\n")
    assert "| catch-up | epiworld-v1-version | 00 | 100 | 2, 3 | 1.50 |" in text

    # A new stage is added after the others; rerunning a stage replaces its rows.
    write_records(runs, "epiworld-v1", 1.0)
    assert [row["stage"] for row in stage("3d-2")] == ["catch-up", "3d-2"]
    rows = stage("catch-up")
    assert [row["stage"] for row in rows] == ["3d-2", "catch-up"]
    assert float(rows[1]["simulate_ratio"]) == 1.0
    capsys.readouterr()


def test_records_from_two_runs_are_not_compared(tmp_path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setattr(history, "RUNS_DIR", runs)
    write_records(runs, "epiworld", 1.0, environment="a" * 64)
    write_records(runs, "epiworld-v1", 1.0, environment="b" * 64)
    with pytest.raises(SystemExit, match="more than one run"):
        history.stage_rows("catch-up")
