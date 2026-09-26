#!/usr/bin/env python3
"""Resource-conscious, resumable orchestration for the ABM benchmark."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any

from scripts.generate_network import ensure_network


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.toml"
CACHE_DIR = ROOT / "cache"
RESULTS_DIR = ROOT / "results"
RUNNER_FORMAT_VERSION = 2
DIST_NAMES = {
    "covasim": "covasim", "EoN": "EoN", "epydemic": "epydemic", "epiworldpy": "epiworldpy",
}
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
    return (
        [CONFIG_PATH, ROOT / "run.py", ROOT / scenario / "scenario.toml"]
        + runners
        + sorted((ROOT / "scripts").glob("*.py"))
    )


def source_hash(scenario: str) -> str:
    digest = hashlib.sha256()
    for path in source_paths(scenario):
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def engine_versions(engines: list[str], scenarios: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for engine in engines:
        if engine in DIST_NAMES:
            versions[engine] = dist_version(DIST_NAMES[engine])
        elif engine == "epiworldR":
            completed = subprocess.run(
                ["Rscript", "--vanilla", "-e", "cat(as.character(packageVersion('epiworldR')))"] ,
                check=True,
                text=True,
                capture_output=True,
            )
            versions[engine] = completed.stdout.strip()
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
            versions[engine] = found.pop()
    return versions


def fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def valid_cache(path: Path, expected_fingerprint: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value.get("status") == "ok" and value.get("fingerprint") == expected_fingerprint
    except (OSError, ValueError):
        return False


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
    if task["engine"] == "epiworldR":
        return ["Rscript", "--vanilla", str(runner_dir / "epiworld.R"), *common]
    if task["engine"] == "ixa":
        return [str(ixa_binary(task["scenario"])), *common]
    if task["engine"] == "epiworld":
        return [str(epiworld_binary(task["scenario"])), *common]
    return [
        sys.executable,
        str(runner_dir / "python_engines.py"),
        "--engine", task["engine"],
        *common,
    ]


def run_task(task: dict[str, Any], env: dict[str, str]) -> tuple[str, bool, str]:
    label = f"{task['scenario']} {task['engine']} n={task['n']} replicate={task['replicate']}"
    completed = subprocess.run(
        task_command(task), cwd=ROOT, env=env, text=True, capture_output=True
    )
    if completed.returncode == 0 and valid_cache(Path(task["output"]), task["fingerprint"]):
        return label, True, completed.stdout.strip()
    message = completed.stderr.strip() or completed.stdout.strip() or "runner produced no diagnostic"
    return label, False, message


def collect_results() -> int:
    records: list[dict[str, Any]] = []
    # Records live at cache/results/<scenario>/<engine>/...; anything else is
    # left over from the single-scenario layout and is ignored.
    for path in sorted((CACHE_DIR / "results").glob("scenario_*/**/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "ok":
                scenario = path.relative_to(CACHE_DIR / "results").parts[0]
                records.append(record | {"scenario": scenario})
        except (OSError, ValueError):
            continue
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "results.csv"
    fields = [
        "scenario", "engine", "engine_version", "n", "days", "replicate", "seed",
        "network_sha256", "network_edges", "mean_degree", "target_r0",
        "transmission_multiplier",
        "setup_seconds", "simulate_seconds", "total_seconds",
        "final_susceptible", "final_exposed", "final_infected",
        "final_hospitalized", "final_recovered", "peak_hospitalized",
        # Scenario-specific outcomes; blank for scenarios that do not report them.
        "vaccinated", "vaccine_protected",
        "fingerprint", "timestamp_utc",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    # The report reads scenario names and parameters from here, so R needs no
    # TOML parser.
    scenarios = {}
    for scenario in discover_scenarios():
        scenario_config = load_scenario(scenario)
        scenarios[scenario] = scenario_config["scenario"] | {
            "parameters": scenario_parameters(scenario_config),
            "calibration": scenario_config.get("calibration", {}),
        }
    atomic_json(RESULTS_DIR / "scenarios.json", scenarios)
    return len(records)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = tomllib.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    study, network, resources = config["study"], config["network"], config["resources"]
    available = discover_scenarios()
    scenarios = args.scenarios or available
    unknown = sorted(set(scenarios) - set(available))
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {', '.join(unknown)}")
    engines = args.engines or list(study["engines"])
    unknown = sorted(set(engines) - set(study["engines"]))
    if unknown:
        raise SystemExit(f"Unknown engine(s): {', '.join(unknown)}")
    scenario_configs = {scenario: load_scenario(scenario) for scenario in scenarios}
    for scenario, scenario_config in scenario_configs.items():
        # Calibration keys are looked up as "<engine>_transmission_multiplier_<n>".
        # A key whose prefix is not an engine name would silently fall back to a
        # multiplier of 1.0, so reject it here instead.
        stray = sorted(
            key for key in scenario_config.get("calibration", {})
            if (match := re.fullmatch(r"(.+)_transmission_multiplier_\d+", key))
            and match.group(1) not in set(study["engines"])
        )
        if stray:
            raise SystemExit(
                f"{scenario}: calibration key(s) do not match any engine name: "
                + ", ".join(stray)
            )
    profile = study if args.profile == "full" else (study | config["smoke"])
    population_sizes = args.sizes or list(profile["population_sizes"])
    replicate_count = args.replicates or int(profile["replicates"])
    if replicate_count < 1:
        raise SystemExit("--replicates must be at least one")
    workers = resolve_workers(args.workers, resources)

    if shutil.which("Rscript") is None and "epiworldR" in engines:
        raise SystemExit("Rscript is required for the epiworldR runner")
    if "ixa" in engines:
        for scenario in scenarios:
            if not ixa_binary(scenario).is_file():
                raise SystemExit(
                    f"ixa runner not built at {ixa_binary(scenario)}; run `make setup`"
                )
    if "epiworld" in engines:
        for scenario in scenarios:
            if not epiworld_binary(scenario).is_file():
                raise SystemExit(
                    f"epiworld runner not built at {epiworld_binary(scenario)}; run `make setup`"
                )
    versions = engine_versions(engines, scenarios)
    host_platform = f"{platform.system()}-{platform.machine()}"
    tasks: list[dict[str, Any]] = []
    cached = 0

    for fallback_index, n in enumerate(population_sizes):
        size_index = (
            list(study["population_sizes"]).index(n)
            if n in study["population_sizes"]
            else fallback_index
        )
        edge_path, network_metadata = ensure_network(
            CACHE_DIR,
            int(n),
            int(network["mean_degree"]),
            float(network["rewire_probability"]),
            int(network["seed"]) + size_index,
        )
        for scenario, scenario_config in scenario_configs.items():
            code_hash = source_hash(scenario)
            calibration = scenario_config.get("calibration", {})
            parameters = scenario_parameters(scenario_config)
            if "initial_infected" in profile and "initial_infected" in parameters:
                parameters["initial_infected"] = int(profile["initial_infected"])
            for engine in engines:
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
                    output = (
                        CACHE_DIR / "results" / scenario / engine / f"n{n}"
                        / f"replicate-{replicate:03d}.json"
                    )
                    if not args.force and valid_cache(output, task_fingerprint):
                        cached += 1
                        continue
                    tasks.append(identity | {
                        "network": str(edge_path),
                        "output": str(output),
                        "fingerprint": task_fingerprint,
                    })

    print(
        f"Profile={args.profile}; scenarios={','.join(scenarios)}; "
        f"engines={','.join(engines)}; "
        f"workers={workers}; cached={cached}; pending={len(tasks)}",
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
    env.update({
        "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
        "RCPP_PARALLEL_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
    })
    failures: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_task, task, env): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            label, ok, message = future.result()
            print(f"[{index}/{len(tasks)}] {'ok' if ok else 'FAILED'} {label}", flush=True)
            if not ok:
                failures.append((label, message))

    count = collect_results()
    manifest = {
        "profile": args.profile,
        "requested_scenarios": scenarios,
        "requested_engines": engines,
        "cached_before_run": cached,
        "executed": len(tasks),
        "failures": len(failures),
        "result_rows_available": count,
        "workers": workers,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "engine_versions": versions,
    }
    atomic_json(RESULTS_DIR / "run-manifest.json", manifest)
    if failures:
        for label, message in failures:
            print(f"\n{label}:\n{message}", file=sys.stderr)
        return 1
    print(f"Wrote {count} cached rows to {RESULTS_DIR / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
