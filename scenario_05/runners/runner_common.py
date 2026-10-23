"""Plumbing shared by scenario 00's Python runners: argument parsing,
edge-list reading, and result writing. Each run_<engine>.py holds one
engine's model and calls main().
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np


# Memory at the timing boundaries, in bytes, merged into the record by main().
# Off Linux, where /proc does not exist, every value is None. Garbage the
# interpreter has not collected yet counts as resident: no collection is forced,
# as it would change the timings.
MEMORY: dict[str, int | None] = {}
MEMORY_FIELDS = (
    "rss_baseline_bytes", "rss_after_read_bytes", "rss_after_setup_bytes",
    "peak_rss_setup_bytes", "peak_rss_simulate_bytes",
)
_peak_reset = False


def memory_status() -> tuple[int | None, int | None]:
    """VmRSS and VmHWM (its high-water mark) of this process, in bytes."""
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            fields = dict(line.split(":", 1) for line in stream if line.startswith(("VmRSS", "VmHWM")))
    except OSError:
        return None, None
    kib = lambda key: int(fields[key].split()[0]) * 1024 if key in fields else None
    return kib("VmRSS"), kib("VmHWM")


def start_simulate() -> float:
    """Record memory after setup, reset the high-water mark, and start the
    simulate timer. Both reads happen before the timer starts."""
    global _peak_reset
    MEMORY["rss_after_setup_bytes"], MEMORY["peak_rss_setup_bytes"] = memory_status()
    try:
        # Writing 5 resets VmHWM to the current RSS (Linux 4.0 and later).
        with open("/proc/self/clear_refs", "w", encoding="ascii") as stream:
            stream.write("5")
        _peak_reset = True
    except OSError:
        _peak_reset = False
    return perf_counter()


def stop_simulate(started: float) -> float:
    """Stop the simulate timer, then record the peak since start_simulate()."""
    elapsed = perf_counter() - started
    # Without a reset the high-water mark would include setup, so it is left out.
    MEMORY["peak_rss_simulate_bytes"] = memory_status()[1] if _peak_reset else None
    return elapsed


def run_measured(command: list[str], **kwargs) -> tuple[subprocess.CompletedProcess, int | None]:
    """subprocess.run(..., text=True, capture_output=True), plus the peak RSS of
    the command in bytes, or None off Linux or without GNU time.

    On Linux, ru_maxrss of a process includes the memory its parent held when it
    forked, here this runner with its edge list. GNU time forks the command from
    its own image, under a megabyte, and reports the command's ru_maxrss.
    """
    gnu_time = shutil.which("time") if sys.platform.startswith("linux") else None
    if gnu_time is None:
        return subprocess.run(command, text=True, capture_output=True, **kwargs), None
    with tempfile.TemporaryDirectory(prefix="maxrss-") as directory:
        report = Path(directory) / "maxrss"
        completed = subprocess.run(
            [gnu_time, "-f", "%M", "-o", str(report), *command],
            text=True, capture_output=True, **kwargs,
        )
        # A failed command adds a "Command exited with ..." line before it.
        lines = report.read_text(encoding="utf-8").split() if report.exists() else []
    completed.args = command
    return completed, int(lines[-1]) * 1024 if lines and lines[-1].isdigit() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--network-sha256", required=True)
    parser.add_argument("--network-edges", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mean-degree", type=float, required=True)
    parser.add_argument("--target-r0", type=float, required=True)
    parser.add_argument("--initial-infected", type=int, required=True)
    parser.add_argument("--latent-days", type=float, required=True)
    parser.add_argument("--infectious-days", type=float, required=True)
    parser.add_argument("--hospitalization-probability", type=float, required=True)
    parser.add_argument("--hospital-days", type=float, required=True)
    parser.add_argument("--transmission-multiplier", type=float, default=1.0)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_edges(path: Path) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    with gzip.open(path, "rt", encoding="ascii") as stream:
        next(stream)
        for line in stream:
            source, target = line.rstrip().split("\t")
            sources.append(int(source))
            targets.append(int(target))
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def atomic_write(path: Path, value: dict) -> None:
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


def main(engine: str, run: Callable[[argparse.Namespace, np.ndarray, np.ndarray], dict], preload: tuple[str, ...]) -> None:
    args = parse_args()
    # Match the R runner's timing boundary: interpreter and engine imports are
    # process startup costs and are excluded from setup/simulation timings.
    for module in preload:
        __import__(module)
    MEMORY["rss_baseline_bytes"] = memory_status()[0]
    total_started = perf_counter()
    source, target = read_edges(args.network)
    if len(source) != args.network_edges:
        raise ValueError(f"edge count mismatch: expected {args.network_edges}, read {len(source)}")
    read_elapsed = perf_counter() - total_started
    MEMORY["rss_after_read_bytes"] = memory_status()[0]
    result = run(args, source, target)
    total_elapsed = perf_counter() - total_started
    # Engines that must re-read their own input files (FRED) report that time
    # here, so it is counted as reading like this edge-file parsing.
    read_elapsed += result.pop("engine_read_seconds", 0.0)
    setup_elapsed = total_elapsed - result["simulate_seconds"]
    record = {
        "status": "ok",
        "engine": engine,
        "engine_version": args.engine_version,
        "n": args.n,
        "days": args.days,
        "replicate": args.replicate,
        "seed": args.seed,
        "network_sha256": args.network_sha256,
        "network_edges": args.network_edges,
        "mean_degree": args.mean_degree,
        "target_r0": args.target_r0,
        "transmission_multiplier": args.transmission_multiplier,
        "read_seconds": read_elapsed,
        "setup_seconds": setup_elapsed,
        "total_seconds": total_elapsed,
        "fingerprint": args.fingerprint,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **{field: MEMORY.get(field) for field in MEMORY_FIELDS},
        **result,
    }
    if sum(record[f"final_{state}"] for state in ("susceptible", "exposed", "infected", "hospitalized", "recovered")) != args.n:
        raise RuntimeError("final compartment counts do not sum to population size")
    atomic_write(args.output, record)
