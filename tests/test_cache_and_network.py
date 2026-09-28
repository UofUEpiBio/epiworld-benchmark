from __future__ import annotations

import json
import tomllib

import run
from run import (
    PYTHON_RUNNERS,
    ROOT,
    CONFIG_PATH,
    discover_scenarios,
    engines_for_scenario,
    epiworld_binary,
    fingerprint,
    ixa_binary,
    load_scenario,
    scenario_parameters,
    size_plan,
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
    engines = (
        "covasim", "starsim", "EoN", "epydemic", "epiworldR", "epiworldpy", "epiworld", "ixa",
        "individual",
    )
    for engine in engines:
        command = task_command(base_task(engine=engine))
        index = command.index("--transmission-multiplier")
        assert command[index + 1] == "0.71"


def test_each_python_engine_has_its_own_runner() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for scenario in discover_scenarios():
        engines = engines_for_scenario(config, load_scenario(scenario))
        for engine, script in PYTHON_RUNNERS.items():
            if engine not in engines:
                continue
            command = task_command(base_task(engine=engine, scenario=scenario))
            path = ROOT / scenario / "runners" / script
            assert command[1] == str(path)
            assert path.is_file(), path
            assert f'main("{engine}", ' in path.read_text()


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
            # The Python runners share their argument parser.
            "python": (runners / "runner_common.py").read_text(),
            "R": (runners / "epiworld.R").read_text(),
            "individual": (runners / "individual.R").read_text(),
            "ixa": (runners / "ixa" / "src" / "main.rs").read_text(),
            "C++": (runners / "epiworld" / "main.cpp").read_text(),
        }
        for key in scenario_parameters(load_scenario(scenario)):
            assert f"--{key.replace('_', '-')}" in sources["python"], (scenario, key)
            assert f'"{key}"' in sources["R"], (scenario, key)
            assert f'"{key}"' in sources["individual"], (scenario, key)
            assert f'"{key}"' in sources["C++"], (scenario, key)
            assert f"{key}:" in sources["ixa"], (scenario, key)


def test_every_runner_records_the_edge_reading_time() -> None:
    for scenario in discover_scenarios():
        runners = ROOT / scenario / "runners"
        for path in (
            runners / "runner_common.py", runners / "epiworld.R", runners / "individual.R",
            runners / "epiworld" / "main.cpp", runners / "ixa" / "src" / "main.rs",
        ):
            assert "read_seconds" in path.read_text(), path


def test_r_engines_run_their_own_script() -> None:
    for engine, script in (("epiworldR", "epiworld.R"), ("individual", "individual.R")):
        command = task_command(base_task(engine=engine))
        assert command[:3] == ["Rscript", "--vanilla", str(ROOT / "scenario_00" / "runners" / script)]


def test_scenario_designs_replace_the_study_sizes_and_replicates() -> None:
    config = {
        "study": {"population_sizes": [10, 100], "replicates": 5},
        "smoke": {"population_sizes": [3], "replicates": 1},
    }
    designs = {"scenario_00": {}, "scenario_03": {"population_sizes": [1000], "replicates": 2}}
    scenarios = ["scenario_00", "scenario_03"]
    assert size_plan(config, "full", scenarios, designs) == [
        (10, 0, ["scenario_00"], 5),
        (100, 1, ["scenario_00"], 5),
        (1000, 2, ["scenario_03"], 2),
    ]
    # The smoke profile ignores [design].
    assert size_plan(config, "smoke", scenarios, designs) == [(3, 0, scenarios, 1)]
    # A size keeps its index (network and seeds) however it is requested, and
    # --sizes only restricts a scenario with a [design].
    assert size_plan(config, "full", ["scenario_03"], designs) == [(1000, 2, ["scenario_03"], 2)]
    assert size_plan(config, "full", scenarios, designs, sizes=[1000], replicates=1) == [
        (1000, 2, scenarios, 1)
    ]
    assert size_plan(config, "full", ["scenario_03"], designs, sizes=[100]) == []


def test_scenario_03_is_scenario_00_at_one_million_agents() -> None:
    designs = run.scenario_designs()
    assert designs["scenario_00"] == {}
    assert designs["scenario_03"] == {
        "population_sizes": [1000000], "replicates": 20, "workers": 1
    }
    assert scenario_parameters(load_scenario("scenario_03")) == scenario_parameters(
        load_scenario("scenario_00")
    )
    for runner in ("epiworld.R", "individual.R", "runner_common.py", "epiworld/main.cpp"):
        assert (ROOT / "scenario_03" / "runners" / runner).read_text() == (
            ROOT / "scenario_00" / "runners" / runner
        ).read_text(), runner


def test_scenarios_are_discovered() -> None:
    assert discover_scenarios()[:3] == ["scenario_00", "scenario_01", "scenario_02"]


def test_source_paths_cover_only_the_scenario_runners() -> None:
    for scenario in discover_scenarios():
        paths = {path.relative_to(ROOT).as_posix() for path in source_paths(scenario)}
        assert f"{scenario}/scenario.toml" in paths
        assert f"{scenario}/runners/ixa/src/main.rs" in paths
        assert f"{scenario}/runners/ixa/Cargo.toml" in paths
        assert f"{scenario}/runners/ixa/Cargo.lock" in paths
        assert f"{scenario}/runners/epiworld.R" in paths
        assert f"{scenario}/runners/individual.R" in paths
        assert f"{scenario}/runners/epiworld/main.cpp" in paths
        assert "config.toml" not in paths and "run.py" not in paths
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


def test_daily_series_are_medians_across_replicates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run, "RESULTS_DIR", tmp_path)
    base = {"scenario": "scenario_02", "engine": "ixa", "n": 10}
    run.write_daily_series([
        base | {"daily_incidence": [1, 2], "reproductive_number": [1.0, None, 0.0]},
        base | {"daily_incidence": [3, 4], "reproductive_number": [2.0, None, 1.0]},
        base | {"daily_incidence": [5, 9], "reproductive_number": [6.0, 2.0, None]},
        # Records from scenarios without daily series are skipped.
        {"scenario": "scenario_01", "engine": "ixa", "n": 10},
    ])
    lines = (tmp_path / "daily.csv").read_text().splitlines()
    assert lines == [
        "scenario,engine,n,day,median_incidence,median_reproductive_number",
        "scenario_02,ixa,10,0,,2.0",
        "scenario_02,ixa,10,1,3,2.0",
        "scenario_02,ixa,10,2,4,0.5",
    ]
