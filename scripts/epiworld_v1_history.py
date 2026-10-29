#!/usr/bin/env python3
"""Add a stage of the epiworld 1.0.0 tracking to the history.

Reads the published records of the two C++ engines, epiworld (0.x) and
epiworld-v1 (1.0.0 in progress), which `make v1-stage` has just rerun in one
invocation. It writes their medians per scenario and size, and their ratio, to
results/epiworld-v1-history.csv under the stage's name (replacing that stage's
rows if it is rerun), and rewrites the table in docs/epiworld-v1.md. The run's
environment record goes to results/epiworld-v1-environments/STAGE.json, as
`make v1-stage` then restores the published records in results/environments/.

Usage: scripts/epiworld_v1_history.py STAGE
"""

from __future__ import annotations

import csv
from datetime import date
import json
from pathlib import Path
import re
import statistics
import sys


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "runs"
HISTORY = ROOT / "results" / "epiworld-v1-history.csv"
ENVIRONMENTS = ROOT / "results" / "environments"
STAGE_ENVIRONMENTS = ROOT / "results" / "epiworld-v1-environments"
DOC = ROOT / "docs" / "epiworld-v1.md"
ENGINES = ("epiworld", "epiworld-v1")
# Medians over replicates, for each engine.
MEASURES = ("simulate_seconds", "build_seconds", "peak_mib", "attack_rate")
FIELDS = [
    "stage", "date", "epiworld_version", "epiworld_v1_version", "environment_id",
    "scenario", "n", "replicates",
    *(f"{measure}_{suffix}" for measure in MEASURES for suffix in ("0x", "v1")),
    "simulate_ratio", "build_ratio", "peak_ratio",
]
BEGIN, END = "<!-- history:begin -->", "<!-- history:end -->"


def measures(record: dict) -> dict[str, float]:
    """The measures of one replicate, as the reports define them."""
    peaks = [
        record.get(key) for key in
        ("peak_rss_setup_bytes", "peak_rss_simulate_bytes", "process_peak_rss_bytes")
    ]
    peaks = [peak for peak in peaks if peak is not None]
    return {
        "simulate_seconds": record["simulate_seconds"],
        # Setup without reading the edge list: building the model.
        "build_seconds": record["setup_seconds"] - record["read_seconds"],
        "peak_mib": max(peaks) / 2**20 if peaks else float("nan"),
        "attack_rate": 1 - record["final_susceptible"] / record["n"],
    }


def stage_rows(stage: str) -> list[dict]:
    records: dict[tuple[str, int], dict[str, list[dict]]] = {}
    for engine in ENGINES:
        for path in sorted(RUNS_DIR.glob(f"scenario_*/{engine}/n*/replicate-*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") == "ok":
                key = (path.relative_to(RUNS_DIR).parts[0], int(record["n"]))
                records.setdefault(key, {}).setdefault(engine, []).append(record)
    rows = []
    for (scenario, n), by_engine in sorted(records.items()):
        if set(by_engine) != set(ENGINES):
            raise SystemExit(f"{scenario} n={n}: records of only {sorted(by_engine)}")
        everything = by_engine["epiworld"] + by_engine["epiworld-v1"]
        # One invocation runs both engines, in one environment.
        environments = {record.get("environment_id") for record in everything}
        if len(environments) != 1 or None in environments:
            raise SystemExit(
                f"{scenario} n={n}: the two engines' records come from more than one run; "
                "rerun them together with `make v1-stage`"
            )
        counts = {len(by_engine[engine]) for engine in ENGINES}
        if len(counts) != 1:
            raise SystemExit(f"{scenario} n={n}: the engines have different replicate counts")
        row = {
            "stage": stage,
            "date": date.today().isoformat(),
            "epiworld_version": by_engine["epiworld"][0]["engine_version"],
            "epiworld_v1_version": by_engine["epiworld-v1"][0]["engine_version"],
            "environment_id": environments.pop()[:12],
            "scenario": scenario,
            "n": n,
            "replicates": counts.pop(),
        }
        for engine, suffix in zip(ENGINES, ("0x", "v1")):
            values = [measures(record) for record in by_engine[engine]]
            for measure in MEASURES:
                row[f"{measure}_{suffix}"] = statistics.median(value[measure] for value in values)
        for ratio, measure in (
            ("simulate_ratio", "simulate_seconds"), ("build_ratio", "build_seconds"),
            ("peak_ratio", "peak_mib"),
        ):
            row[ratio] = row[f"{measure}_v1"] / row[f"{measure}_0x"]
        rows.append(row)
    if not rows:
        raise SystemExit(f"No records of {' and '.join(ENGINES)} in {RUNS_DIR}")
    return rows


def write_history(rows: list[dict]) -> list[dict]:
    """Replaces the stage's rows, keeping the stages in the order they were added."""
    history = []
    if HISTORY.is_file():
        with HISTORY.open(encoding="utf-8", newline="") as stream:
            history = [row for row in csv.DictReader(stream) if row["stage"] != rows[0]["stage"]]
    history += rows
    with HISTORY.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in history:
            writer.writerow({
                key: f"{value:.6g}" if isinstance(value, float) else value
                for key, value in row.items()
            })
    return history


def table(history: list[dict]) -> str:
    lines = [
        "| Stage | epiworld-v1 | Scenario | Agents | Simulate (s): 0.x, 1.0 | Ratio "
        "| Build (s): 0.x, 1.0 | Ratio | Peak (MiB): 0.x, 1.0 | Ratio | Attack rate: 0.x, 1.0 |",
        "|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for row in history:
        value = {key: float(row[key]) for key in FIELDS[8:]}
        lines.append(
            f"| {row['stage']} | {row['epiworld_v1_version']} | {row['scenario'][-2:]} "
            f"| {int(row['n']):,} "
            f"| {value['simulate_seconds_0x']:.3g}, {value['simulate_seconds_v1']:.3g} "
            f"| {value['simulate_ratio']:.2f} "
            f"| {value['build_seconds_0x']:.3g}, {value['build_seconds_v1']:.3g} "
            f"| {value['build_ratio']:.2f} "
            f"| {value['peak_mib_0x']:.0f}, {value['peak_mib_v1']:.0f} "
            f"| {value['peak_ratio']:.2f} "
            f"| {value['attack_rate_0x']:.3f}, {value['attack_rate_v1']:.3f} |"
        )
    return "\n".join(lines)


def save_environment(stage: str, environment_id: str) -> None:
    """The run's environment record (the same for every scenario, but for the
    counts of replicates), one per scenario."""
    records = {
        path.stem: record for path in sorted(ENVIRONMENTS.glob("scenario_*.json"))
        if (record := json.loads(path.read_text(encoding="utf-8")))
        .get("environment_id", "").startswith(environment_id)
    }
    if not records:
        raise SystemExit(f"No environment record in {ENVIRONMENTS} has id {environment_id}")
    STAGE_ENVIRONMENTS.mkdir(parents=True, exist_ok=True)
    (STAGE_ENVIRONMENTS / f"{stage}.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    if len(sys.argv) != 2 or not re.fullmatch(r"[\w.-]+", sys.argv[1]):
        raise SystemExit(__doc__ + "\nSTAGE is letters, digits, '.', '-' and '_'.")
    stage = sys.argv[1]
    rows = stage_rows(stage)
    save_environment(stage, rows[0]["environment_id"])
    history = write_history(rows)
    text = DOC.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.DOTALL)
    if not pattern.search(text):
        raise SystemExit(f"{DOC} has no {BEGIN} ... {END} block")
    DOC.write_text(
        pattern.sub(lambda _: f"{BEGIN}\n{table(history)}\n{END}", text), encoding="utf-8"
    )
    print(f"Wrote {len(history)} rows to {HISTORY.relative_to(ROOT)} and the table in "
          f"{DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
