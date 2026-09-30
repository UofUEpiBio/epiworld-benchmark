from __future__ import annotations

import json
import os
import sys

import run
from run import TaskOutcome, merge_into_record, write_environment_records
from scripts import environment
from scripts.environment import collect_environment, environment_id


def outcome(scenario: str, ok: bool = True, started="2026-01-01T00:00:00+00:00") -> TaskOutcome:
    return TaskOutcome(f"{scenario} ixa", ok, "", scenario, started, "2026-01-01T01:00:00+00:00")


def test_environment_id_is_stable_and_ignores_run_details() -> None:
    first = collect_environment({"ixa": "3.1.0"}, workers=1, profile="smoke")
    second = collect_environment({"ixa": "3.1.0"}, workers=4, profile="full")
    assert first["environment_id"] == second["environment_id"] == environment_id(first)
    # Repository state and run settings are not part of the identity.
    changed = first | {"repository": {"commit": "x", "dirty": True}, "run": {"workers": 9}}
    assert environment_id(changed) == first["environment_id"]


def test_environment_id_changes_with_what_determines_performance() -> None:
    base = collect_environment({"ixa": "3.1.0"}, workers=1, profile="smoke")
    other_versions = collect_environment({"ixa": "3.2.0"}, workers=1, profile="smoke")
    assert other_versions["environment_id"] != base["environment_id"]
    for key, value in (
        ("hardware", {"cpu_model": "other"}),
        ("os", {"container": True}),
        ("container_image", "sha256:other"),
        ("toolchains", {"r": "R version 0"}),
        ("pins", {"FRED_SHA": "0" * 40}),
    ):
        assert environment_id(base | {key: value}) != base["environment_id"], key


def test_unreadable_fields_are_null(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(environment.subprocess, "run", missing)
    monkeypatch.setattr(environment, "_read", lambda path: None)
    monkeypatch.delenv("BENCHMARK_IMAGE_ID", raising=False)
    for name in environment.PIN_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    found = collect_environment({}, workers=1, profile="smoke")
    assert found["container_image"] is None
    assert found["repository"] == {"commit": None, "dirty": None}
    assert found["hardware"]["cgroup_cpu_limit"] is None
    assert found["hardware"]["cgroup_memory_limit_bytes"] is None
    assert found["toolchains"]["julia"] is None and found["toolchains"]["quarto"] is None
    assert found["pins"]["FRED_SHA"] is None
    assert len(found["environment_id"]) == 64


def test_cgroup_limits_are_parsed() -> None:
    assert environment.parse_cpu_max("200000 100000\n") == 2.0
    assert environment.parse_cpu_max("max 100000\n") is None
    assert environment.parse_cpu_max(None) is None
    assert environment.parse_memory_max("8589934592\n") == 8589934592
    assert environment.parse_memory_max("max\n") is None


def test_image_id_and_pins_come_from_the_container_environment(monkeypatch) -> None:
    monkeypatch.setenv("BENCHMARK_IMAGE_ID", "sha256:abc")
    monkeypatch.setenv("EPIWORLD_SHA", "a" * 40)
    monkeypatch.setenv("FRED_SHA", "b" * 40)
    found = collect_environment({}, workers=1, profile="smoke")
    assert found["container_image"] == "sha256:abc"
    assert found["pins"]["EPIWORLD_SHA"] == "a" * 40
    assert found["pins"]["FRED_SHA"] == "b" * 40


def test_scenarios_without_executed_tasks_keep_their_record(tmp_path) -> None:
    environment_record = {"environment_id": "new", "hardware": {}}
    for scenario in ("scenario_00", "scenario_01"):
        (tmp_path / f"{scenario}.json").write_text('{"environment_id": "old"}\n')

    # A fully cached run executes nothing, so nothing is rewritten.
    assert write_environment_records(
        environment_record, [], {"scenario_00": 5, "scenario_01": 5}, tmp_path
    ) == []
    for scenario in ("scenario_00", "scenario_01"):
        assert (tmp_path / f"{scenario}.json").read_text() == '{"environment_id": "old"}\n'

    # Forcing one scenario updates only its record.
    written = write_environment_records(
        environment_record,
        [outcome("scenario_01", started="2026-01-01T00:10:00+00:00"), outcome("scenario_01", ok=False)],
        {"scenario_00": 5},
        tmp_path,
    )
    assert written == ["scenario_01"]
    assert (tmp_path / "scenario_00.json").read_text() == '{"environment_id": "old"}\n'
    record = json.loads((tmp_path / "scenario_01.json").read_text())
    assert record["environment_id"] == "new"
    assert record["scenario"] == "scenario_01"
    assert record["executed"] == 2 and record["failures"] == 1 and record["cached"] == 0
    assert record["started_utc"] == "2026-01-01T00:00:00+00:00"
    assert record["finished_utc"] == "2026-01-01T01:00:00+00:00"


def test_merge_into_record_adds_fields_and_keeps_the_rest(tmp_path) -> None:
    path = tmp_path / "replicate-001.json"
    path.write_text(json.dumps({"status": "ok", "fingerprint": "f", "simulate_seconds": 0.25}))
    merge_into_record(path, {"environment_id": "e"})
    assert json.loads(path.read_text()) == {
        "status": "ok", "fingerprint": "f", "simulate_seconds": 0.25, "environment_id": "e",
    }
    assert run.valid_cache(path, "f", None)


def test_run_task_completes_the_record_its_runner_wrote(tmp_path, monkeypatch) -> None:
    output = tmp_path / "replicate-001.json"
    # A real child process, so wait4() runs: it holds about 64 MB and writes
    # its record as a runner would.
    script = (
        "import json, sys; block = bytearray(64 * 2**20); block[::4096] = b'x' * len(block[::4096]); "
        f"json.dump({{'status': 'ok', 'fingerprint': 'f'}}, open({str(output)!r}, 'w')); print('done')"
    )
    monkeypatch.setattr(run, "task_command", lambda task: [sys.executable, "-c", script])
    task = {
        "scenario": "scenario_00", "engine": "ixa", "n": 10, "replicate": 1,
        "output": str(output), "fingerprint": "f", "environment_id": "e",
    }
    result = run.run_task(task, dict(os.environ))
    assert result.ok and result.message == "done" and result.scenario == "scenario_00"
    record = json.loads(output.read_text())
    assert record["environment_id"] == "e"
    assert record["format_version"] == run.RUNNER_FORMAT_VERSION
    # In bytes on every platform.
    assert 64 * 2**20 <= record["process_peak_rss_bytes"] < 2**31
    assert run.valid_cache(output, "f")


def test_run_task_reports_a_failed_runner(tmp_path, monkeypatch) -> None:
    command = [sys.executable, "-c", "import sys; sys.exit('model diverged')"]
    monkeypatch.setattr(run, "task_command", lambda task: command)
    task = {
        "scenario": "scenario_00", "engine": "ixa", "n": 10, "replicate": 1,
        "output": str(tmp_path / "missing.json"), "fingerprint": "f", "environment_id": "e",
    }
    result = run.run_task(task, dict(os.environ))
    assert not result.ok and result.message == "model diverged"


def test_placeholder_cpu_model_falls_back_to_the_vendor(monkeypatch) -> None:
    lscpu = "Architecture: aarch64\nVendor ID: Apple\nModel name: -\nSocket(s): -\n"
    monkeypatch.setattr(environment.sys, "platform", "linux")
    monkeypatch.setattr(environment, "_read", lambda path: "processor : 0\nCPU part : 0x000\n")
    monkeypatch.setattr(environment, "_run", lambda command: lscpu)
    assert environment.cpu_model() == "Apple (model not reported)"
    monkeypatch.setattr(environment, "_run", lambda command: lscpu.replace(": -", ": M3"))
    assert environment.cpu_model() == "M3"


def test_host_cpu_comes_from_the_makefile_in_a_container(monkeypatch) -> None:
    monkeypatch.setattr(environment, "in_container", lambda: True)
    monkeypatch.delenv("BENCHMARK_HOST_CPU", raising=False)
    assert environment.hardware()["host_cpu_model"] is None
    monkeypatch.setenv("BENCHMARK_HOST_CPU", "Apple M3 Pro")
    assert environment.hardware()["host_cpu_model"] == "Apple M3 Pro"
    monkeypatch.setattr(environment, "in_container", lambda: False)
    monkeypatch.delenv("BENCHMARK_HOST_CPU")
    assert environment.hardware()["host_cpu_model"] == environment.cpu_model()
