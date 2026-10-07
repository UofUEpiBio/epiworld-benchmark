#!/usr/bin/env python3
"""Resource-conscious, resumable orchestration for the ABM benchmark."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import tomllib
from typing import Any, NamedTuple

from scripts.environment import RUNNER_ENV, collect_environment, pins
from scripts.generate_network import ensure_network_from_config


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.toml"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
ENVIRONMENTS_DIR = RESULTS_DIR / "environments"
# One JSON record per full-profile replicate, committed with the results, so any
# clone skips what has been published and runs from several machines combine by
# copying files. results.csv and daily.csv are derived from them. Smoke records
# are throwaway and stay in the untracked cache.
RUNS_DIR = RESULTS_DIR / "runs"
# Bump when the contents of a cache record change. Records carry it as
# format_version, and valid_cache() rejects older ones: identity alone is not
# enough, as accept_published_results matches published fingerprints without it.
# Version 3 added the memory measures.
RUNNER_FORMAT_VERSION = 3
DIST_NAMES = {
    "covasim": "covasim", "EoN": "EoN", "epydemic": "epydemic", "epiworldpy": "epiworldpy",
    "starsim": "starsim",
}
# One script per Python engine in each scenario's runners/ folder. They are
# not named after the engine, which the script would then shadow on import.
PYTHON_RUNNERS = {
    "covasim": "run_covasim.py", "EoN": "run_eon.py", "epydemic": "run_epydemic.py",
    "epiworldpy": "run_epiworldpy.py", "starsim": "run_starsim.py", "FRED": "run_FRED.py",
}
# R engines and their runner in each scenario's runners/ folder.
R_RUNNERS = {"epiworldR": "epiworld.R", "individual": "individual.R", "ABM": "abm.R"}
R_PACKAGES = {"epiworldR": "epiworldR", "individual": "individual", "ABM": "ABM"}
JULIA_RUNNERS = {"Agents.jl": "agents.jl"}
# Scenario tables whose keys are forwarded to every runner as --kebab-case flags.
PARAMETER_TABLES = ("disease", "intervention")


def discover_scenarios() -> list[str]:
    return sorted(path.parent.name for path in ROOT.glob("scenario_*/scenario.toml"))


def load_scenario(scenario: str) -> dict[str, Any]:
    path = ROOT / scenario / "scenario.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


def scenario_parameters(scenario_config: dict[str, Any]) -> dict[str, Any]:
    """Model parameters forwarded to the runners, in a stable order."""
    parameters: dict[str, Any] = {}
    for table in PARAMETER_TABLES:
        for key, value in scenario_config.get(table, {}).items():
            if key in parameters:
                raise SystemExit(f"Parameter {key!r} is defined in more than one table")
            parameters[key] = value
    return parameters


def ixa_dir(scenario: str) -> Path:
    return ROOT / scenario / "runners" / "ixa"


def ixa_binary(scenario: str) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", ixa_dir(scenario) / "target"))
    return target / "release" / f"ixa-{scenario.replace('_', '-')}"


def epiworld_binary(scenario: str) -> Path:
    """The scenario's compiled epiworld (C++) runner; `make setup` builds it."""
    build = Path(
        os.environ.get("EPIWORLD_BUILD_DIR", ROOT / scenario / "runners" / "epiworld" / "build")
    )
    return build / f"epiworld-{scenario.replace('_', '-')}"


def fred_home() -> Path:
    """FRED is a single shared build, not compiled per scenario like epiworld/ixa."""
    return Path(os.environ.get("FRED_HOME", ROOT / ".deps" / "fred"))


def fred_binary() -> Path:
    return fred_home() / "bin" / "FRED"


def dist_version(name: str) -> str:
    """Installed version, plus the commit for packages installed from git."""
    version = importlib.metadata.version(name)
    direct_url = importlib.metadata.distribution(name).read_text("direct_url.json")
    commit = json.loads(direct_url or "{}").get("vcs_info", {}).get("commit_id")
    return f"{version}+g{commit[:7]}" if commit else version


def ixa_version(scenario: str) -> str:
    lock = tomllib.loads((ixa_dir(scenario) / "Cargo.lock").read_text(encoding="utf-8"))
    return next(package["version"] for package in lock["package"] if package["name"] == "ixa")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_into_record(path: Path, fields: dict[str, Any]) -> None:
    """Add `fields` to a cached replicate record, which its runner has written."""
    record = json.loads(path.read_text(encoding="utf-8"))
    atomic_json(path, record | fields)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def source_paths(scenario: str) -> list[Path]:
    """Files whose contents define a scenario's results, excluding build output.

    The scenario's README.md and code_regions.yml are documentation and report
    inputs, so editing them does not invalidate cached results.
    """
    runner_dir = ROOT / scenario / "runners"
    runners = [
        path for path in sorted(runner_dir.rglob("*"))
        if path.is_file()
        and not {"target", "build"} & set(path.relative_to(runner_dir).parts)
        and "__pycache__" not in path.parts
    ]
    # Only scenario-owned model inputs belong in this hash. The input network
    # has its own SHA-256 identity; orchestrator-only changes must not discard
    # completed simulations. Changes to result identity instead bump
    # RUNNER_FORMAT_VERSION.
    return [ROOT / scenario / "scenario.toml"] + runners


def source_hash(scenario: str) -> str:
    digest = hashlib.sha256()
    for path in source_paths(scenario):
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def published_fingerprint_registry() -> dict[tuple[Any, ...], str]:
    registry = RESULTS_DIR / "results.csv"
    if not registry.is_file():
        return {}
    values = {}
    with registry.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["scenario"], row["engine"], row["engine_version"], int(row["n"]),
                int(row["days"]), int(row["replicate"]), int(row["seed"]),
                row["network_sha256"], int(row["network_edges"]),
                float(row["transmission_multiplier"]),
            )
            values[key] = row["fingerprint"]
    return values


def compatible_fingerprints(identity: dict[str, Any], scenario_config: dict[str, Any]) -> set[str]:
    """Current hash plus the exact hash in the committed results registry.

    The registry is used only by scenarios that explicitly opt in. It lets the
    source-hash boundary be narrowed without invalidating already published
    runs; identifying inputs must still match the task being planned.
    """
    expected = {fingerprint(identity)}
    if not scenario_config.get("cache", {}).get("accept_published_results", False):
        return expected
    key = (
        identity["scenario"], identity["engine"], str(identity["engine_version"]), identity["n"],
        identity["days"], identity["replicate"], identity["seed"],
        identity["network_sha256"], identity["network_edges"],
        identity["transmission_multiplier"],
    )
    if published := published_fingerprint_registry().get(key):
        expected.add(published)
    return expected


def engines_for_scenario(config: dict[str, Any], scenario_config: dict[str, Any]) -> list[str]:
    return list(scenario_config.get("design", {}).get("engines", config["study"]["engines"]))


# Engines built from a pinned commit whose version number does not change
# with it; the commit is appended so that moving a pin changes the fingerprint.
PIN_OF = {"epiworldR": "EPIWORLDR_SHA"}


def with_commit(version: str, commit: str | None) -> str:
    return f"{version}+g{commit[:7]}" if commit else version


def engine_versions(engines: list[str], scenarios: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for engine in engines:
        if engine in DIST_NAMES:
            versions[engine] = dist_version(DIST_NAMES[engine])
        elif engine in R_PACKAGES:
            completed = subprocess.run(
                [
                    "Rscript", "--vanilla", "-e",
                    f"cat(as.character(packageVersion('{R_PACKAGES[engine]}')))",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            versions[engine] = with_commit(completed.stdout.strip(), pins().get(PIN_OF.get(engine)))
        elif engine == "ixa":
            found = {ixa_version(scenario) for scenario in scenarios}
            if len(found) != 1:
                raise SystemExit(f"Scenarios pin different ixa versions: {sorted(found)}")
            versions[engine] = found.pop()
        elif engine == "epiworld":
            # Each runner reports the epiworld headers it was compiled against.
            found = {
                subprocess.run(
                    [str(epiworld_binary(scenario)), "--version"],
                    check=True, text=True, capture_output=True,
                ).stdout.strip()
                for scenario in scenarios
            }
            if len(found) != 1:
                raise SystemExit(
                    f"Scenarios were built with different epiworld versions: {sorted(found)}"
                )
            versions[engine] = with_commit(found.pop(), pins()["EPIWORLD_SHA"])
        elif engine == "Agents.jl":
            versions[engine] = subprocess.run(
                [
                    "julia", f"--project={ROOT / 'julia'}", "--startup-file=no", "-e",
                    "using Agents; print(Base.pkgversion(Agents))",
                ],
                check=True, text=True, capture_output=True,
            ).stdout.strip()
        elif engine == "FRED":
            # FRED has no --version flag; the pinned commit is written to
            # COMMIT next to the binary when it is built (see the Makefile
            # and the container image).
            release = (fred_home() / "VERSION").read_text(encoding="utf-8").strip()
            commit = (fred_home() / "COMMIT").read_text(encoding="utf-8").strip()
            versions[engine] = f"{release}+g{commit[:7]}"
    return versions


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def valid_cache(
    path: Path,
    expected_fingerprint: str | set[str],
    format_version: int | None = RUNNER_FORMAT_VERSION,
) -> bool:
    """A successful record with a matching fingerprint and, unless
    `format_version` is None (a runner's record before the orchestrator adds
    its fields), the current record format."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = (
            {expected_fingerprint} if isinstance(expected_fingerprint, str) else expected_fingerprint
        )
        return (
            value.get("status") == "ok"
            and value.get("fingerprint") in expected
            and (format_version is None or value.get("format_version") == format_version)
        )
    except (OSError, ValueError):
        return False


# On Linux, ru_maxrss of a process includes the memory its parent held when it
# forked, which would put the orchestrator's own size into every runner's peak.
# GNU time forks the runner from its own image, under a megabyte, and reports
# the runner's ru_maxrss. Without it the process peak is not recorded.
GNU_TIME = shutil.which("time") if sys.platform.startswith("linux") else None


def run_measured(
    command: list[str], **kwargs: Any
) -> tuple[subprocess.CompletedProcess, int | None]:
    """subprocess.run(..., text=True, capture_output=True), plus the peak RSS
    in bytes of the command and of every descendant it waited for."""
    if GNU_TIME is None:
        return subprocess.run(command, text=True, capture_output=True, **kwargs), None
    fd, report = tempfile.mkstemp(prefix="maxrss-")
    os.close(fd)
    try:
        completed = subprocess.run(
            [GNU_TIME, "-f", "%M", "-o", report, *command],
            text=True, capture_output=True, **kwargs,
        )
        # A failed command adds a "Command exited with ..." line before it.
        lines = Path(report).read_text(encoding="utf-8").split()
    finally:
        os.unlink(report)
    completed.args = command
    return completed, int(lines[-1]) * 1024 if lines and lines[-1].isdigit() else None


def resolve_workers(cli_value: int | None, resources: dict[str, Any]) -> int:
    """Harness concurrency: --workers, else $N_THREADS, else the config default.

    $N_THREADS is deliberately not part of source_hash(), so tuning concurrency
    does not invalidate cached results the way editing config.toml would.
    """
    if cli_value is not None:
        requested = cli_value
    elif (raw := os.environ.get("N_THREADS", "").strip()):
        try:
            requested = int(raw)
        except ValueError:
            raise SystemExit(f"N_THREADS must be an integer, got {raw!r}") from None
    else:
        requested = int(resources["workers"])
    ceiling = min(int(resources["max_workers"]), os.cpu_count() or 1)
    return max(1, min(requested, ceiling))


def flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def task_command(task: dict[str, Any]) -> list[str]:
    common = [
        "--network", task["network"],
        "--network-sha256", task["network_sha256"],
        "--network-edges", str(task["network_edges"]),
        "--n", str(task["n"]),
        "--days", str(task["days"]),
        "--replicate", str(task["replicate"]),
        "--seed", str(task["seed"]),
        "--mean-degree", str(task["mean_degree"]),
    ]
    for key, value in task["parameters"].items():
        common += [flag(key), str(value)]
    common += [
        "--transmission-multiplier", str(task["transmission_multiplier"]),
        "--fingerprint", task["fingerprint"],
        "--engine-version", task["engine_version"],
        "--output", task["output"],
    ]
    runner_dir = ROOT / task["scenario"] / "runners"
    if task["engine"] in R_RUNNERS:
        return ["Rscript", "--vanilla", str(runner_dir / R_RUNNERS[task["engine"]]), *common]
    if task["engine"] in JULIA_RUNNERS:
        return [
            "julia", f"--project={ROOT / 'julia'}", "--startup-file=no",
            str(runner_dir / JULIA_RUNNERS[task["engine"]]), *common,
        ]
    if task["engine"] == "ixa":
        return [str(ixa_binary(task["scenario"])), *common]
    if task["engine"] == "epiworld":
        return [str(epiworld_binary(task["scenario"])), *common]
    return [sys.executable, str(runner_dir / PYTHON_RUNNERS[task["engine"]]), *common]


class TaskOutcome(NamedTuple):
    label: str
    ok: bool
    message: str
    scenario: str
    started_utc: str
    finished_utc: str


def run_task(task: dict[str, Any], env: dict[str, str]) -> TaskOutcome:
    label = f"{task['scenario']} {task['engine']} n={task['n']} replicate={task['replicate']}"
    started = utc_now()
    completed, peak_rss = run_measured(task_command(task), cwd=ROOT, env=env)
    output = Path(task["output"])
    ok = completed.returncode == 0 and valid_cache(output, task["fingerprint"], None)
    message = completed.stdout.strip()
    if not ok:
        message = completed.stderr.strip() or message or "runner produced no diagnostic"
    else:
        # The runner does not know where it runs, nor its own peak memory as
        # the operating system saw it; the orchestrator adds both to its record.
        # The runners reset their high-water mark when the simulate timer
        # starts, which lowers ru_maxrss too, so this peak is an independent
        # check of peak_rss_simulate_bytes and whatever follows, not of the
        # whole run.
        try:
            merge_into_record(output, {
                "environment_id": task["environment_id"],
                "format_version": RUNNER_FORMAT_VERSION,
                "process_peak_rss_bytes": peak_rss,
            })
        except (OSError, ValueError) as error:
            ok, message = False, f"could not complete {output}: {error}"
    return TaskOutcome(label, ok, message, task["scenario"], started, utc_now())


def write_environment_records(
    environment: dict[str, Any],
    outcomes: list[TaskOutcome],
    cached: dict[str, int],
    directory: Path,
) -> list[str]:
    """One record per scenario that executed a task, in `directory`.

    A scenario whose replicates were all cached keeps the record of the run
    that produced them. Returns the scenarios written.
    """
    written = []
    for scenario in sorted({outcome.scenario for outcome in outcomes}):
        own = [outcome for outcome in outcomes if outcome.scenario == scenario]
        atomic_json(directory / f"{scenario}.json", environment | {
            "scenario": scenario,
            "started_utc": min(outcome.started_utc for outcome in own),
            "finished_utc": max(outcome.finished_utc for outcome in own),
            "executed": len(own),
            "failures": sum(not outcome.ok for outcome in own),
            "cached": cached.get(scenario, 0),
        })
        written.append(scenario)
    return written


def write_daily_series(records: list[dict[str, Any]]) -> None:
    """Median daily incidence and reproductive number across replicates.

    Records list incidence for days 1 to `days` and the reproductive number
    for days 0 to `days` (day 0 holds the seed cases); a day with no cases has
    no reproductive number and is left out of its median. The per-replicate
    series stay in the cache; only their medians are small enough to publish.
    """
    series: dict[tuple[str, str, int, int], dict[str, list[float]]] = {}
    for record in records:
        if "daily_incidence" not in record:
            continue
        for key, first_day in (("daily_incidence", 1), ("reproductive_number", 0)):
            for day, value in enumerate(record[key], start=first_day):
                cell = series.setdefault(
                    (record["scenario"], record["engine"], record["n"], day),
                    {"daily_incidence": [], "reproductive_number": []},
                )
                if value is not None:
                    cell[key].append(value)
    with (RESULTS_DIR / "daily.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "scenario", "engine", "n", "day", "median_incidence", "median_reproductive_number",
        ])
        for (scenario, engine, n, day), cell in sorted(series.items()):
            median = lambda values: statistics.median(values) if values else ""
            writer.writerow([
                scenario, engine, n, day,
                median(cell["daily_incidence"]), median(cell["reproductive_number"]),
            ])


def report_collected(count: int) -> None:
    if count:
        print(f"Wrote {count} published rows to {RESULTS_DIR / 'results.csv'}")
    else:
        print(f"No published records in {RUNS_DIR}; left {RESULTS_DIR / 'results.csv'} as it was")


def records_dir(profile: str) -> Path:
    """Where a profile's replicate records live: published for the full
    profile, the untracked cache for smoke runs."""
    return RUNS_DIR if profile == "full" else CACHE_DIR / "results"


def record_path(profile: str, scenario: str, engine: str, n: int, replicate: int) -> Path:
    return records_dir(profile) / scenario / engine / f"n{n}" / f"replicate-{replicate:03d}.json"


def collect_results() -> int:
    """Rebuild the derived results files from the published records alone, so
    they are the same on every clone whatever its cache holds."""
    records: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.glob("scenario_*/*/n*/replicate-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "ok":
                scenario = path.relative_to(RUNS_DIR).parts[0]
                records.append(record | {"scenario": scenario})
        except (OSError, ValueError):
            continue
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "results.csv"
    if not records and output.is_file():
        # Nothing published yet (a smoke-only clone, or results/runs/ not yet
        # populated): keep the committed results rather than emptying them.
        return 0
    fields = [
        "scenario", "engine", "engine_version", "n", "days", "replicate", "seed",
        "network_sha256", "network_edges", "mean_degree", "target_r0",
        "transmission_multiplier",
        "read_seconds", "setup_seconds", "simulate_seconds", "total_seconds",
        "final_susceptible", "final_exposed", "final_infected",
        "final_hospitalized", "final_recovered", "peak_hospitalized",
        # Scenario-specific outcomes; blank for scenarios that do not report them.
        "vaccinated", "vaccine_protected",
        "extract_seconds", "transmissions",
        "transitions_se", "transitions_ei", "transitions_ih", "transitions_ir", "transitions_hr",
        "rss_baseline_bytes", "rss_after_read_bytes", "rss_after_setup_bytes",
        "peak_rss_setup_bytes", "peak_rss_simulate_bytes", "process_peak_rss_bytes",
        "fingerprint", "environment_id", "timestamp_utc",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    write_daily_series(records)
    # The report reads scenario names and parameters from here, so R needs no
    # TOML parser.
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    designs = scenario_designs()
    scenarios = {}
    for scenario in discover_scenarios():
        scenario_config = load_scenario(scenario)
        scenarios[scenario] = scenario_config["scenario"] | {
            "parameters": scenario_parameters(scenario_config),
            "calibration": scenario_config.get("calibration", {}),
            "engines": engines_for_scenario(config, scenario_config),
            "network": scenario_config.get("network", config["network"]),
            # Concurrency cap from [design], if any (see main()).
            "workers": designs[scenario].get("workers"),
            # The full design: replicates per engine at each population size.
            "design": [
                {"n": n, "replicates": replicates}
                for n, _, _, replicates in size_plan(config, "full", [scenario], designs)
            ],
        }
    atomic_json(RESULTS_DIR / "scenarios.json", scenarios)
    return len(records)


def scenario_designs() -> dict[str, dict[str, Any]]:
    """Each scenario's [design] table, which overrides [study] sizes and replicates."""
    return {scenario: load_scenario(scenario).get("design", {}) for scenario in discover_scenarios()}


def size_plan(
    config: dict[str, Any],
    profile: str,
    scenarios: list[str],
    designs: dict[str, dict[str, Any]],
    sizes: list[int] | None = None,
    replicates: int | None = None,
) -> list[tuple[int, int, list[str], int]]:
    """(n, size index, scenarios, replicates) for each population size to run.

    In the full profile, a scenario runs at the population sizes and replicate
    count in its scenario.toml [design] table, or else at those in [study].
    The smoke profile ignores [design]. --sizes replaces the sizes of a
    scenario without a [design] and restricts the sizes of one with it;
    --replicates replaces every count. A size's index, which picks its network
    and seeds, is its position in [study] population_sizes, then in the
    [design] tables of `designs`, however the size is requested.
    """
    study = config["study"]
    profile_config = study if profile == "full" else (study | config["smoke"])
    design_sizes = [int(n) for n in study["population_sizes"]]
    for design in designs.values():
        design_sizes += [
            int(n) for n in design.get("population_sizes", []) if int(n) not in design_sizes
        ]
    requested = [int(n) for n in sizes] if sizes else None
    cells: dict[tuple[int, int], list[str]] = {}
    for scenario in scenarios:
        design = designs.get(scenario, {}) if profile == "full" else {}
        scenario_sizes = [
            int(n) for n in design.get("population_sizes", profile_config["population_sizes"])
        ]
        if requested is not None:
            scenario_sizes = (
                [n for n in requested if n in scenario_sizes]
                if "population_sizes" in design else requested
            )
        count = replicates or int(design.get("replicates", profile_config["replicates"]))
        for n in scenario_sizes:
            cells.setdefault((n, count), []).append(scenario)
    order = requested or [int(n) for n in profile_config["population_sizes"]]
    return [
        (
            n,
            design_sizes.index(n) if n in design_sizes else (order.index(n) if n in order else 0),
            size_scenarios,
            count,
        )
        for (n, count), size_scenarios in sorted(cells.items())
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--scenarios", nargs="+", help="subset of scenario folders (default: all scenario_*)"
    )
    parser.add_argument("--engines", nargs="+", help="subset of configured engines")
    parser.add_argument("--sizes", nargs="+", type=int, help="override profile population sizes")
    parser.add_argument("--replicates", type=int, help="override profile replicate count")
    parser.add_argument(
        "--workers", type=int,
        help="harness concurrency; overrides $N_THREADS (hard-capped by config)",
    )
    parser.add_argument("--force", action="store_true", help="ignore matching cached results")
    parser.add_argument("--dry-run", action="store_true", help="show work without running models")
    parser.add_argument(
        "--collect", action="store_true",
        help="only rebuild results.csv, daily.csv, and scenarios.json from results/runs/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.collect:
        report_collected(collect_results())
        return 0
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    study, resources = config["study"], config["resources"]
    available = discover_scenarios()
    scenarios = args.scenarios or available
    unknown = sorted(set(scenarios) - set(available))
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {', '.join(unknown)}")
    scenario_configs = {scenario: load_scenario(scenario) for scenario in scenarios}
    available_engines = list(study["engines"])
    requested_engines = args.engines
    unknown = sorted(set(requested_engines or []) - set(available_engines))
    if unknown:
        raise SystemExit(f"Unknown engine(s): {', '.join(unknown)}")
    scenario_engine_map = {
        scenario: [
            engine for engine in engines_for_scenario(config, scenario_config)
            if requested_engines is None or engine in requested_engines
        ]
        for scenario, scenario_config in scenario_configs.items()
    }
    engines = list(dict.fromkeys(
        engine for scenario in scenarios for engine in scenario_engine_map[scenario]
    ))
    for scenario, scenario_config in scenario_configs.items():
        # Calibration keys are looked up as "<engine>_transmission_multiplier_<n>".
        # A key whose prefix is not an engine name would silently fall back to a
        # multiplier of 1.0, so reject it here instead.
        stray = sorted(
            key for key in scenario_config.get("calibration", {})
            if (match := re.fullmatch(r"(.+)_transmission_multiplier_\d+", key))
            and match.group(1) not in set(available_engines)
        )
        if stray:
            raise SystemExit(
                f"{scenario}: calibration key(s) do not match any engine name: "
                + ", ".join(stray)
            )
    profile = study if args.profile == "full" else (study | config["smoke"])
    if args.replicates is not None and args.replicates < 1:
        raise SystemExit("--replicates must be at least one")
    designs = scenario_designs()
    plan = size_plan(config, args.profile, scenarios, designs, args.sizes, args.replicates)
    workers = resolve_workers(args.workers, resources)

    if shutil.which("Rscript") is None and set(R_RUNNERS) & set(engines):
        raise SystemExit("Rscript is required for the R runners")
    if shutil.which("julia") is None and set(JULIA_RUNNERS) & set(engines):
        raise SystemExit("Julia is required for the Agents.jl runner; use the container")
    if "ixa" in engines:
        for scenario in scenarios:
            if "ixa" not in scenario_engine_map[scenario]:
                continue
            if not ixa_binary(scenario).is_file():
                raise SystemExit(
                    f"ixa runner not built at {ixa_binary(scenario)}; run `make setup`"
                )
    if "epiworld" in engines:
        for scenario in scenarios:
            if "epiworld" not in scenario_engine_map[scenario]:
                continue
            if not epiworld_binary(scenario).is_file():
                raise SystemExit(
                    f"epiworld runner not built at {epiworld_binary(scenario)}; run `make setup`"
                )
    if "FRED" in engines and not fred_binary().is_file():
        raise SystemExit(f"FRED runner not built at {fred_binary()}; run `make setup`")
    versions = engine_versions(engines, scenarios)
    environment = collect_environment(versions, workers, args.profile)
    host_platform = f"{platform.system()}-{platform.machine()}"
    tasks: list[dict[str, Any]] = []
    cached: dict[str, int] = {}

    for n, size_index, size_scenarios, replicate_count in plan:
        for scenario in size_scenarios:
            scenario_config = scenario_configs[scenario]
            network_config = scenario_config.get("network", config["network"])
            edge_path, network_metadata = ensure_network_from_config(
                CACHE_DIR, int(n), network_config, size_index
            )
            code_hash = source_hash(scenario)
            calibration = scenario_config.get("calibration", {})
            parameters = scenario_parameters(scenario_config)
            if "initial_infected" in profile and "initial_infected" in parameters:
                parameters["initial_infected"] = int(profile["initial_infected"])
            for engine in scenario_engine_map[scenario]:
                for replicate in range(1, replicate_count + 1):
                    # Seeds do not depend on the scenario, so scenarios can be
                    # compared replicate by replicate.
                    seed = int(study["base_seed"]) + size_index * 100_000 + replicate
                    identity = {
                        "format_version": RUNNER_FORMAT_VERSION,
                        # Timings are host-specific, so records from a native run
                        # and a container run are never mixed.
                        "platform": host_platform,
                        "source_hash": code_hash,
                        "scenario": scenario,
                        "engine": engine,
                        "engine_version": versions[engine],
                        "n": int(n),
                        "days": int(profile["days"]),
                        "replicate": replicate,
                        "seed": seed,
                        "network_sha256": network_metadata["sha256"],
                        "network_edges": network_metadata["edges"],
                        "mean_degree": network_metadata["mean_degree_observed"],
                        "parameters": parameters,
                        "transmission_multiplier": float(
                            calibration.get(f"{engine}_transmission_multiplier_{int(n)}", 1.0)
                        ),
                    }
                    task_fingerprint = fingerprint(identity)
                    output = record_path(args.profile, scenario, engine, int(n), replicate)
                    expected_fingerprints = compatible_fingerprints(identity, scenario_config)
                    if not args.force and valid_cache(output, expected_fingerprints):
                        cached[scenario] = cached.get(scenario, 0) + 1
                        continue
                    tasks.append(identity | {
                        "environment_id": environment["environment_id"],
                        "network": str(edge_path),
                        "output": str(output),
                        "fingerprint": task_fingerprint,
                    })

    print(
        f"Profile={args.profile}; scenarios={','.join(scenarios)}; "
        f"engines={','.join(engines)}; "
        f"workers={workers}; cached={sum(cached.values())}; pending={len(tasks)}",
        flush=True,
    )
    if workers > 1 and tasks:
        # Concurrent replicates contend for memory bandwidth and, on hybrid
        # CPUs, for performance cores. The effect is uneven across engines, so
        # timings from a parallel run are not comparable to sequential ones.
        print(
            f"NOTE: running {workers} replicates concurrently; timings are "
            "comparable only against runs at the same concurrency. High worker "
            "counts inflate simulate_seconds unevenly across engines - see the "
            "resource policy in setup.md.",
            file=sys.stderr, flush=True,
        )
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.update(RUNNER_ENV)
    # A scenario's [design] can cap concurrency for its own runs in the full
    # profile: scenario_03's million-agent runs compete for memory when
    # several run at once, so they run one at a time.
    groups: dict[int, list[dict[str, Any]]] = {}
    for task in tasks:
        design = designs.get(task["scenario"], {}) if args.profile == "full" else {}
        groups.setdefault(min(workers, int(design.get("workers", workers))), []).append(task)
    failures: list[tuple[str, str]] = []
    outcomes: list[TaskOutcome] = []
    index = 0
    for group_workers, group in groups.items():
        with ThreadPoolExecutor(max_workers=group_workers) as pool:
            futures = {pool.submit(run_task, task, env): task for task in group}
            for future in as_completed(futures):
                index += 1
                outcome = future.result()
                outcomes.append(outcome)
                print(
                    f"[{index}/{len(tasks)}] {'ok' if outcome.ok else 'FAILED'} {outcome.label}",
                    flush=True,
                )
                if not outcome.ok:
                    failures.append((outcome.label, outcome.message))

    count = collect_results()
    write_environment_records(environment, outcomes, cached, ENVIRONMENTS_DIR)
    manifest = {
        "profile": args.profile,
        "requested_scenarios": scenarios,
        "requested_engines": engines,
        "cached_before_run": sum(cached.values()),
        "executed": len(tasks),
        "failures": len(failures),
        "result_rows_available": count,
        "workers": workers,
        "python": platform.python_version(),
    }
    atomic_json(RESULTS_DIR / "run-manifest.json", manifest)
    if failures:
        for label, message in failures:
            print(f"\n{label}:\n{message}", file=sys.stderr)
        return 1
    report_collected(count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
