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
from scripts.generate_network import ensure_network, sha256_edges, sha256_file


def test_fingerprint_is_order_independent() -> None:
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_cache_requires_success_and_matching_fingerprint(tmp_path) -> None:
    path = tmp_path / "result.json"
    current = {"format_version": run.RUNNER_FORMAT_VERSION}
    path.write_text(json.dumps({"status": "ok", "fingerprint": "right"} | current))
    assert valid_cache(path, "right")
    assert not valid_cache(path, "wrong")
    path.write_text(json.dumps({"status": "failed", "fingerprint": "right"} | current))
    assert not valid_cache(path, "right")


def test_cache_rejects_records_in_an_older_format(tmp_path, monkeypatch) -> None:
    """A published fingerprint must not bring back a record without memory data."""
    path = tmp_path / "replicate-001.json"
    path.write_text(json.dumps({"status": "ok", "fingerprint": "published"}))
    assert not valid_cache(path, {"current", "published"})
    path.write_text(json.dumps({"status": "ok", "fingerprint": "published", "format_version": 2}))
    assert not valid_cache(path, {"current", "published"})
    # A runner's own record, before the orchestrator completes it, is checked
    # without the format.
    assert valid_cache(path, "published", None)

    # The same through compatible_fingerprints(), as a scenario that accepts
    # published results sees it.
    identity = {
        "scenario": "scenario_00", "engine": "ixa", "engine_version": "3.1.0", "n": 10,
        "days": 5, "replicate": 1, "seed": 1, "network_sha256": "x", "network_edges": 50,
        "transmission_multiplier": 1.0, "format_version": run.RUNNER_FORMAT_VERSION,
    }
    key = (
        "scenario_00", "ixa", "3.1.0", 10, 5, 1, 1, "x", 50, 1.0,
    )
    monkeypatch.setattr(run, "published_fingerprint_registry", lambda: {key: "published"})
    expected = run.compatible_fingerprints(identity, {"cache": {"accept_published_results": True}})
    assert "published" in expected
    assert not valid_cache(path, expected)
    path.write_text(json.dumps({
        "status": "ok", "fingerprint": "published", "format_version": run.RUNNER_FORMAT_VERSION,
    }))
    assert valid_cache(path, expected)


def test_network_cache_is_deterministic_and_verified(tmp_path) -> None:
    first_path, first = ensure_network(tmp_path, 100, 10, 0.05, 42)
    second_path, second = ensure_network(tmp_path, 100, 10, 0.05, 42)
    assert first_path == second_path
    assert first == second
    assert first["edges"] == 500
    assert first["mean_degree_observed"] == 10
    assert first["sha256"] == sha256_edges(first_path)


def test_network_identity_does_not_depend_on_the_compressor(tmp_path) -> None:
    """Another zlib build compresses the same edges to different bytes; the
    network, and so every fingerprint, must keep its checksum."""
    import gzip

    path, metadata = ensure_network(tmp_path, 100, 10, 0.05, 42)
    edges = gzip.decompress(path.read_bytes())
    path.write_bytes(gzip.compress(edges, compresslevel=1, mtime=0))
    assert sha256_file(path) != metadata["sha256"]
    assert sha256_edges(path) == metadata["sha256"]
    # The cached network is still accepted, not regenerated.
    assert ensure_network(tmp_path, 100, 10, 0.05, 42)[1] == metadata


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
        "individual", "ABM", "EpiModel", "FRED", "Agents.jl",
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
            "ABM": (runners / "abm.R").read_text(),
            "EpiModel": (runners / "epimodel.R").read_text(),
            "ixa": (runners / "ixa" / "src" / "main.rs").read_text(),
            "C++": (runners / "epiworld" / "main.cpp").read_text(),
            "Julia": (runners / "agents.jl").read_text(),
        }
        for key in scenario_parameters(load_scenario(scenario)):
            assert f"--{key.replace('_', '-')}" in sources["python"], (scenario, key)
            assert f'"{key}"' in sources["R"], (scenario, key)
            assert f'"{key}"' in sources["individual"], (scenario, key)
            assert f'"{key}"' in sources["ABM"], (scenario, key)
            assert f'"{key}"' in sources["EpiModel"], (scenario, key)
            assert f'"{key}"' in sources["C++"], (scenario, key)
            assert f'"{key}"' in sources["Julia"], (scenario, key)
            assert f"{key}:" in sources["ixa"], (scenario, key)


def test_every_runner_records_the_edge_reading_time() -> None:
    for scenario in discover_scenarios():
        runners = ROOT / scenario / "runners"
        for path in (
            runners / "runner_common.py", runners / "epiworld.R", runners / "individual.R",
            runners / "abm.R", runners / "epimodel.R", runners / "epiworld" / "main.cpp", runners / "ixa" / "src" / "main.rs",
            runners / "agents.jl",
        ):
            assert "read_seconds" in path.read_text(), path


def test_every_runner_records_memory_at_the_timing_boundaries() -> None:
    fields = (
        "rss_baseline_bytes", "rss_after_read_bytes", "rss_after_setup_bytes",
        "peak_rss_setup_bytes", "peak_rss_simulate_bytes",
    )
    for scenario in discover_scenarios():
        runners = ROOT / scenario / "runners"
        for path in (
            runners / "runner_common.py", runners / "epiworld.R", runners / "individual.R",
            runners / "abm.R", runners / "epimodel.R", runners / "epiworld" / "main.cpp", runners / "ixa" / "src" / "main.rs",
            runners / "agents.jl", runners / "run_FRED.py",
        ):
            source = path.read_text()
            assert all(field in source for field in fields), path
            if path.name != "run_FRED.py":
                assert "/proc/self/clear_refs" in source, path
        # Each Python engine resets the peak where its simulate timer starts.
        for engine, script in PYTHON_RUNNERS.items():
            if engine != "FRED":
                source = (runners / script).read_text()
                assert "start_simulate()" in source and "stop_simulate(started)" in source, script


def test_r_engines_run_their_own_script() -> None:
    for engine, script in (("epiworldR", "epiworld.R"), ("individual", "individual.R"), ("ABM", "abm.R"),
                           ("EpiModel", "epimodel.R")):
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
    for runner in ("epiworld.R", "individual.R", "abm.R", "epimodel.R", "runner_common.py", "epiworld/main.cpp"):
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
        assert f"{scenario}/runners/abm.R" in paths
        assert f"{scenario}/runners/epimodel.R" in paths
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


def test_every_scenario_runs_every_engine() -> None:
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for scenario in discover_scenarios():
        engines = engines_for_scenario(config, load_scenario(scenario))
        assert "FRED" in engines and "Agents.jl" in engines, scenario
        runners = ROOT / scenario / "runners"
        assert (runners / "run_FRED.py").is_file(), scenario
        assert (runners / "agents.jl").is_file(), scenario


def test_moving_an_epiworld_pin_changes_the_recorded_version(monkeypatch) -> None:
    """epiworld and epiworldR report only 0.17.1 whatever their commit, so the
    commit is appended; otherwise a new pin would reuse published results."""
    def fake(command, **kwargs):
        stdout = "0.17.1.0" if command[0] == "Rscript" else "0.17.1\n"
        return run.subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(run.subprocess, "run", fake)
    monkeypatch.setattr(run, "pins", lambda: {"EPIWORLD_SHA": "04c4ad8" + "0" * 33,
                                              "EPIWORLDR_SHA": "fca60f3" + "0" * 33})
    versions = run.engine_versions(["epiworld", "epiworldR", "individual"], ["scenario_00"])
    assert versions == {
        "epiworld": "0.17.1+g04c4ad8", "epiworldR": "0.17.1.0+gfca60f3", "individual": "0.17.1.0",
    }
    # Without a known commit (epiworldR natively), the version stands alone.
    monkeypatch.setattr(run, "pins", lambda: {"EPIWORLD_SHA": None, "EPIWORLDR_SHA": None})
    assert run.engine_versions(["epiworld", "epiworldR"], ["scenario_00"]) == {
        "epiworld": "0.17.1", "epiworldR": "0.17.1.0",
    }
