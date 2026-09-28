"""Plumbing shared by scenario 02's Python runners: argument parsing,
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
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np


# Compartment order of the transition matrices.
STATES = ("S", "E", "I", "H", "R")


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
    parser.add_argument("--vaccine-coverage", type=float, required=True)
    parser.add_argument("--vaccine-efficacy", type=float, required=True)
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


def output_summary(days: int, transmissions: int, matrix: np.ndarray, incidence, reproductive_number) -> dict:
    """Result-record fields summarizing the four outputs. Day 0 holds the seed
    cases, so the transition totals and incidence cover days 1 to `days`."""
    pair = lambda a, b: int(matrix[1:, STATES.index(a), STATES.index(b)].sum())
    return {
        "transmissions": int(transmissions),
        "transitions_se": pair("S", "E"),
        "transitions_ei": pair("E", "I"),
        "transitions_ih": pair("I", "H"),
        "transitions_ir": pair("I", "R"),
        "transitions_hr": pair("H", "R"),
        "daily_incidence": [int(value) for value in incidence[1:days + 1]],
        "reproductive_number": [None if np.isnan(value) else float(value) for value in reproductive_number],
    }


def main(engine: str, run: Callable[[argparse.Namespace, np.ndarray, np.ndarray], dict], preload: tuple[str, ...]) -> None:
    args = parse_args()
    # Match the R runner's timing boundary: interpreter and engine imports are
    # process startup costs and are excluded from setup/simulation timings.
    for module in preload:
        __import__(module)
    total_started = perf_counter()
    source, target = read_edges(args.network)
    if len(source) != args.network_edges:
        raise ValueError(f"edge count mismatch: expected {args.network_edges}, read {len(source)}")
    result = run(args, source, target)
    total_elapsed = perf_counter() - total_started
    setup_elapsed = total_elapsed - result["simulate_seconds"] - result["extract_seconds"]
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
        "setup_seconds": setup_elapsed,
        "total_seconds": total_elapsed,
        "fingerprint": args.fingerprint,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    if sum(record[f"final_{state}"] for state in ("susceptible", "exposed", "infected", "hospitalized", "recovered")) != args.n:
        raise RuntimeError("final compartment counts do not sum to population size")
    atomic_write(args.output, record)
