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
RUNNER_FORMAT_VERSION = 1
DIST_NAMES = {"covasim": "covasim", "EoN": "EoN", "epydemic": "epydemic"}
IXA_DIR = ROOT / "runners" / "ixa"


def ixa_binary() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", IXA_DIR / "target"))
    return target / "release" / "ixa-benchmark"


def ixa_version() -> str:
    lock = tomllib.loads((IXA_DIR / "Cargo.lock").read_text(encoding="utf-8"))
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


def source_paths() -> list[Path]:
    """Files whose contents define the benchmark, excluding build output."""
    runners = [
        path for path in sorted((ROOT / "runners").rglob("*"))
        if path.is_file()
        and "target" not in path.relative_to(ROOT / "runners").parts
        and "__pycache__" not in path.parts
    ]
    return (
        [CONFIG_PATH, ROOT / "run.py"]
        + runners
        + sorted((ROOT / "scripts").glob("*.py"))
    )


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in source_paths():
        if path.is_file():
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def engine_versions(engines: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for engine in engines:
        if engine in DIST_NAMES:
            versions[engine] = importlib.metadata.version(DIST_NAMES[engine])
        elif engine == "epiworldR":
            completed = subprocess.run(
                ["Rscript", "--vanilla", "-e", "cat(as.character(packageVersion('epiworldR')))"] ,
                check=True,
                text=True,
                capture_output=True,
            )
            versions[engine] = completed.stdout.strip()
        elif engine == "ixa":
            versions[engine] = ixa_version()
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
        "--target-r0", str(task["target_r0"]),
        "--initial-infected", str(task["initial_infected"]),
        "--latent-days", str(task["latent_days"]),
        "--infectious-days", str(task["infectious_days"]),
        "--hospitalization-probability", str(task["hospitalization_probability"]),
        "--hospital-days", str(task["hospital_days"]),
        "--transmission-multiplier", str(task["transmission_multiplier"]),
        "--fingerprint", task["fingerprint"],
        "--engine-version", task["engine_version"],
        "--output", task["output"],
    ]
    if task["engine"] == "epiworldR":
        return [
            "Rscript", "--vanilla", str(ROOT / "runners" / "epiworld.R"),
            *common,
        ]
    if task["engine"] == "ixa":
        return [str(ixa_binary()), *common]
    return [
        sys.executable,
        str(ROOT / "runners" / "python_engines.py"),
        "--engine", task["engine"],
        *common,
    ]


def run_task(task: dict[str, Any], env: dict[str, str]) -> tuple[str, bool, str]:
    label = f"{task['engine']} n={task['n']} replicate={task['replicate']}"
    completed = subprocess.run(
        task_command(task), cwd=ROOT, env=env, text=True, capture_output=True
    )
    if completed.returncode == 0 and valid_cache(Path(task["output"]), task["fingerprint"]):
        return label, True, completed.stdout.strip()
    message = completed.stderr.strip() or completed.stdout.strip() or "runner produced no diagnostic"
    return label, False, message


def collect_results() -> int:
    records: list[dict[str, Any]] = []
    for path in sorted((CACHE_DIR / "results").glob("**/*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "ok":
                records.append(record)
        except (OSError, ValueError):
            continue
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "results.csv"
    fields = [
        "engine", "engine_version", "n", "days", "replicate", "seed",
        "network_sha256", "network_edges", "mean_degree", "target_r0",
        "transmission_multiplier",
        "setup_seconds", "simulate_seconds", "total_seconds",
        "final_susceptible", "final_exposed", "final_infected",
        "final_hospitalized", "final_recovered", "peak_hospitalized",
        "fingerprint", "timestamp_utc",
    ]
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
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
    study, network, disease, resources, calibration = (
        config["study"], config["network"], config["disease"], config["resources"],
        config["calibration"],
    )
    engines = args.engines or list(study["engines"])
    unknown = sorted(set(engines) - set(study["engines"]))
    if unknown:
        raise SystemExit(f"Unknown engine(s): {', '.join(unknown)}")
    # Calibration keys are looked up as "<engine>_transmission_multiplier_<n>".
    # A key whose prefix is not an engine name would silently fall back to a
    # multiplier of 1.0, so reject it here instead.
    stray = sorted(
        key for key in calibration
        if (match := re.fullmatch(r"(.+)_transmission_multiplier_\d+", key))
        and match.group(1) not in set(study["engines"])
    )
    if stray:
        raise SystemExit(
            "Calibration key(s) do not match any engine name: " + ", ".join(stray)
        )
    profile = study if args.profile == "full" else (study | config["smoke"])
    population_sizes = args.sizes or list(profile["population_sizes"])
    replicate_count = args.replicates or int(profile["replicates"])
    if replicate_count < 1:
        raise SystemExit("--replicates must be at least one")
    workers = resolve_workers(args.workers, resources)

    if shutil.which("Rscript") is None and "epiworldR" in engines:
        raise SystemExit("Rscript is required for the epiworldR runner")
    if "ixa" in engines and not ixa_binary().is_file():
        raise SystemExit(f"ixa runner not built at {ixa_binary()}; run `make setup`")
    versions = engine_versions(engines)
    code_hash = source_hash()
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
        for engine in engines:
            for replicate in range(1, replicate_count + 1):
                seed = int(study["base_seed"]) + size_index * 100_000 + replicate
                identity = {
                    "format_version": RUNNER_FORMAT_VERSION,
                    # Timings are host-specific, so records from a native run
                    # and a container run are never mixed.
                    "platform": host_platform,
                    "source_hash": code_hash,
                    "engine": engine,
                    "engine_version": versions[engine],
                    "n": int(n),
                    "days": int(profile["days"]),
                    "replicate": replicate,
                    "seed": seed,
                    "network_sha256": network_metadata["sha256"],
                    "network_edges": network_metadata["edges"],
                    "mean_degree": network_metadata["mean_degree_observed"],
                    "target_r0": disease["target_r0"],
                    "initial_infected": int(profile.get("initial_infected", disease["initial_infected"])),
                    "latent_days": disease["latent_days"],
                    "infectious_days": disease["infectious_days"],
                    "hospitalization_probability": disease["hospitalization_probability"],
                    "hospital_days": disease["hospital_days"],
                    "transmission_multiplier": float(
                        calibration.get(f"{engine}_transmission_multiplier_{int(n)}", 1.0)
                    ),
                }
                task_fingerprint = fingerprint(identity)
                output = CACHE_DIR / "results" / engine / f"n{n}" / f"replicate-{replicate:03d}.json"
                if not args.force and valid_cache(output, task_fingerprint):
                    cached += 1
                    continue
                tasks.append(identity | {
                    "network": str(edge_path),
                    "output": str(output),
                    "fingerprint": task_fingerprint,
                })

    print(
        f"Profile={args.profile}; engines={','.join(engines)}; "
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
