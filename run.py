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


def source_hash() -> str:
    digest = hashlib.sha256()
    paths = (
        [CONFIG_PATH, ROOT / "run.py"]
        + sorted((ROOT / "runners").glob("*"))
        + sorted((ROOT / "scripts").glob("*.py"))
    )
    for path in paths:
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
        "--fingerprint", task["fingerprint"],
        "--engine-version", task["engine_version"],
        "--output", task["output"],
    ]
    if task["engine"] == "epiworldR":
        return [
            "Rscript", "--vanilla", str(ROOT / "runners" / "epiworld.R"),
            *common,
            "--transmission-multiplier", str(task["transmission_multiplier"]),
        ]
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
    parser.add_argument("--workers", type=int, help="override safe default (hard-capped by config)")
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
    profile = study if args.profile == "full" else (study | config["smoke"])
    population_sizes = args.sizes or list(profile["population_sizes"])
    replicate_count = args.replicates or int(profile["replicates"])
    if replicate_count < 1:
        raise SystemExit("--replicates must be at least one")
    workers = args.workers if args.workers is not None else int(resources["workers"])
    workers = max(1, min(workers, int(resources["max_workers"]), 2))

    if shutil.which("Rscript") is None and "epiworldR" in engines:
        raise SystemExit("Rscript is required for the epiworldR runner")
    versions = engine_versions(engines)
    code_hash = source_hash()
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
                    "transmission_multiplier": (
                        float(calibration[f"epiworld_transmission_multiplier_{int(n)}"])
                        if engine == "epiworldR"
                        and f"epiworld_transmission_multiplier_{int(n)}" in calibration
                        else 1.0
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
