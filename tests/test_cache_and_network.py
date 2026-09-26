from __future__ import annotations

import json

from run import (
    ROOT,
    discover_scenarios,
    epiworld_binary,
    fingerprint,
    ixa_binary,
    load_scenario,
    scenario_parameters,
    source_hash,
    source_paths,
    task_command,
    valid_cache,
)
from scripts.generate_network import ensure_network, sha256_file


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_cache_requires_success_and_matching_fingerprint(tmp_path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "ok", "fingerprint": "right"}))
    assert valid_cache(path, "right")
    assert not valid_cache(path, "wrong")
    path.write_text(json.dumps({"status": "failed", "fingerprint": "right"}))
    assert not valid_cache(path, "right")


def test_network_cache_is_deterministic_and_verified(tmp_path) -> None:
    first_path, first = ensure_network(tmp_path, 100, 10, 0.05, 42)
    second_path, second = ensure_network(tmp_path, 100, 10, 0.05, 42)
    assert first_path == second_path
    assert first == second
    assert first["edges"] == 500
    assert first["mean_degree_observed"] == 10
    assert first["sha256"] == sha256_file(first_path)


def base_task(**overrides) -> dict:
    return {
        "scenario": "scenario_00",
        "network": "network.tsv.gz",
        "network_sha256": "checksum",
        "network_edges": 50,
        "n": 10,
        "days": 5,
        "replicate": 1,
        "seed": 1,
        "mean_degree": 10.0,
        "parameters": {"target_r0": 2.0, "initial_infected": 1},
        "transmission_multiplier": 0.71,
        "fingerprint": "fingerprint",
        "engine_version": "version",
        "output": "result.json",
    } | overrides


def test_task_command_passes_transmission_multiplier_to_each_runner() -> None:
    for engine in ("covasim", "epiworldR", "epiworldpy", "epiworld", "ixa"):
        command = task_command(base_task(engine=engine))
        index = command.index("--transmission-multiplier")
        assert command[index + 1] == "0.71"


def test_scenario_parameters_become_runner_flags() -> None:
    parameters = scenario_parameters(load_scenario("scenario_01"))
    assert parameters["vaccine_coverage"] == 0.30
    assert parameters["vaccine_efficacy"] == 0.80
    for engine in ("covasim", "epiworldR", "epiworldpy", "epiworld", "ixa"):
        command = task_command(
            base_task(engine=engine, scenario="scenario_01", parameters=parameters)
        )
        assert command[command.index("--vaccine-coverage") + 1] == "0.3"
        assert command[command.index("--latent-days") + 1] == "4.0"
        runner = " ".join(command[:3])
        assert "scenario_01" in runner or "-scenario-01" in runner


def test_runners_accept_exactly_their_scenario_parameters() -> None:
    """Every scenario flag must appear in each runner's source."""
    for scenario in discover_scenarios():
        runners = ROOT / scenario / "runners"
        sources = {
            "python": (runners / "python_engines.py").read_text(),
            "R": (runners / "epiworld.R").read_text(),
            "ixa": (runners / "ixa" / "src" / "main.rs").read_text(),
            "C++": (runners / "epiworld" / "main.cpp").read_text(),
        }
        for key in scenario_parameters(load_scenario(scenario)):
            assert f"--{key.replace('_', '-')}" in sources["python"], (scenario, key)
            assert f'"{key}"' in sources["R"], (scenario, key)
            assert f'"{key}"' in sources["C++"], (scenario, key)
            assert f"{key}:" in sources["ixa"], (scenario, key)


def test_scenarios_are_discovered() -> None:
    assert discover_scenarios()[:2] == ["scenario_00", "scenario_01"]


def test_source_paths_cover_only_the_scenario_runners() -> None:
    for scenario in discover_scenarios():
        paths = {path.relative_to(ROOT).as_posix() for path in source_paths(scenario)}
        assert f"{scenario}/scenario.toml" in paths
        assert f"{scenario}/runners/ixa/src/main.rs" in paths
        assert f"{scenario}/runners/ixa/Cargo.toml" in paths
        assert f"{scenario}/runners/ixa/Cargo.lock" in paths
        assert f"{scenario}/runners/epiworld.R" in paths
        assert f"{scenario}/runners/epiworld/main.cpp" in paths
        assert "config.toml" in paths
        assert not any("/target/" in path or "/build/" in path for path in paths)
        assert not any(
            path.endswith(("README.md", "README.qmd", "code_regions.yml")) or "README_files" in path
            for path in paths
        )
        others = set(discover_scenarios()) - {scenario}
        assert not any(path.split("/")[0] in others for path in paths)


def test_scenarios_have_distinct_source_hashes() -> None:
    assert source_hash("scenario_00") != source_hash("scenario_01")


def test_ixa_binary_is_named_after_the_scenario() -> None:
    assert ixa_binary("scenario_01").name == "ixa-scenario-01"
    manifest = (ROOT / "scenario_01" / "runners" / "ixa" / "Cargo.toml").read_text()
    assert 'name = "ixa-scenario-01"' in manifest


def test_epiworld_binary_is_named_after_the_scenario(monkeypatch) -> None:
    monkeypatch.delenv("EPIWORLD_BUILD_DIR", raising=False)
    binary = epiworld_binary("scenario_01")
    assert binary.name == "epiworld-scenario-01"
    assert binary.parent == ROOT / "scenario_01" / "runners" / "epiworld" / "build"
    monkeypatch.setenv("EPIWORLD_BUILD_DIR", "/opt/epiworld-build")
    assert str(epiworld_binary("scenario_00")) == "/opt/epiworld-build/epiworld-scenario-00"
