#!/usr/bin/env python3
"""Single-replicate FRED runner for benchmark scenario 04.

FRED has no edge-list network primitive comparable to the other engines'
graphs: it is a location-based synthetic-population simulator. This runner
represents the shared network as a synthetic population of single-person
households (so FRED's own household mixing adds no contacts beyond the
network) plus FRED's `Network` group type, whose edges are loaded verbatim
from the benchmark's edge list as `Contact.add_edge` properties.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter

import numpy as np

from runner_common import main


def daily_edge_probability(target_r0: float, degree: float, infectious_days: float) -> float:
    """Per-day, per-edge transmission probability matching R0 over infectious_days.

    FRED's network transmission draws, once a day, an expected `contact_rate *
    degree` successful contacts per infectious neighbour (see
    Network_Transmission::transmission); `contact_rate_for_BENCH` below is set
    to exactly cancel the daily time block, so the condition's
    `transmissibility` property is this function's per-day probability
    directly. Over infectious_days days that compounds to the same per-edge
    lifetime transmission probability the other engines target.
    """
    lifetime_probability = min(0.999, target_r0 / max(1.0, degree - 1.0))
    return 1.0 - (1.0 - lifetime_probability) ** (1.0 / infectious_days)


def write_population(pop_dir: Path, n: int) -> None:
    county = pop_dir / "usa" / "bench" / "00000"
    county.mkdir(parents=True)
    with open(county / "households.txt", "w") as stream:
        stream.write("sp_id\tstcotrbg\trace\thh_income\tlatitude\tlongitude\televation\n")
        stream.writelines(f"{i}\t450830001001\t1\t50000\t34.95\t-81.93\t0\n" for i in range(n))
    with open(county / "people.txt", "w") as stream:
        stream.write("sp_id\tsp_hh_id\tage\tsex\trace\trelate\tschool_id\twork_id\n")
        stream.writelines(f"{i}\t{i}\t30\tM\t1\t0\tX\tX\n" for i in range(n))
    for name, header in (
        ("schools.txt", "sp_id\tstco\tlatitude\tlongitude\televation\n"),
        ("workplaces.txt", "sp_id\tlatitude\tlongitude\televation\n"),
        ("gq_people.txt", "sp_id\tsp_gq_id\tage\tsex\n"),
        ("hospitals.txt", "hosp_id\tworkers\tphysicians\tbeds\tlatitude\tlongitude\televation\n"),
        ("gq.txt", "sp_id\tgq_type\tstcotrbg\tpersons\tlatitude\tlongitude\televation\n"),
    ):
        (county / name).write_text(header)


def write_model(
    work_dir: Path, pop_dir: Path, locations_file: Path, args: argparse.Namespace,
    source: np.ndarray, target: np.ndarray, edge_probability: float,
) -> Path:
    initial_infected = min(args.initial_infected, args.n)
    lines = [
        f"population_directory = {pop_dir}",
        "population_version = bench",
        "country = usa",
        f"locations_file = {locations_file}",
        "start_date = 2020-Jan-01",
        f"days = {args.days}",
        f"seed = {args.seed}",
        "quality_control = 0",
        "enable_health_records = 0",
        "include_network = Contact",
        "Contact.is_undirected = 1",
        "Contact.can_transmit_BENCH = 1",
        # Without an explicit open-hours schedule, Group_Type::get_time_block
        # returns 0 and the network never transmits. One block covering every
        # hour of every day keeps the network open continuously.
        "Contact.starts_at_hour_0_on_weekdays = 24",
        "Contact.starts_at_hour_0_on_weekends = 24",
        # See daily_edge_probability: this rate cancels the 24-hour block so
        # the condition's transmissibility below is a direct daily probability.
        "Contact.contact_rate_for_BENCH = 0.0416666666666667",
        "Contact.deterministic_contacts_for_BENCH = 1",
    ]
    lines.extend(f"Contact.add_edge = {a} {b} 1.0" for a, b in zip(source.tolist(), target.tolist()))
    lines.append(f"""
condition BENCH {{
  states = S E I H R Import
  import_start_state = Import
  transmission_mode = network
  transmissible_networks = Contact
  transmission_network = Contact
  transmissibility = {edge_probability!r}
  exposed_state = E
}}
state BENCH.S {{
  set_sus(BENCH,1)
  wait()
  next()
}}
state BENCH.E {{
  set_sus(BENCH,0)
  wait(24*geometric({args.latent_days!r}))
  next(I)
}}
state BENCH.I {{
  set_trans(BENCH,1)
  wait(24*geometric({args.infectious_days!r}))
  next(H) with prob({args.hospitalization_probability!r})
  default(R)
}}
state BENCH.H {{
  set_trans(BENCH,0)
  wait(24*geometric({args.hospital_days!r}))
  next(R)
}}
state BENCH.R {{
  set_trans(BENCH,0)
  wait()
  next()
}}
# The import_agent (a virtual person, not a member of the population) is
# itself driven through this state machine, so it must never leave Import:
# a next() target here would permanently misclassify it as a real case in
# whatever state it landed in. select_imported_cases (triggered by
# import_count) exposes the real seed cases via exposed_state above.
state BENCH.Import {{
  import_count({initial_infected})
  wait()
  next()
}}
""")
    model_path = work_dir / "model.fred"
    model_path.write_text("\n".join(lines))
    return model_path


def run_fred(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    fred_home = os.environ.get("FRED_HOME")
    if not fred_home:
        raise RuntimeError("FRED_HOME is not set; see `make setup` or the container image")
    fred_binary = Path(fred_home) / "bin" / "FRED"

    with tempfile.TemporaryDirectory(prefix="fred-run-") as tmp:
        work_dir = Path(tmp)
        pop_dir = work_dir / "pop"
        locations_file = work_dir / "locations.txt"
        locations_file.write_text("00000\n")
        write_population(pop_dir, args.n)
        edge_probability = daily_edge_probability(
            args.target_r0, args.mean_degree, args.infectious_days
        ) * args.transmission_multiplier
        model_path = write_model(work_dir, pop_dir, locations_file, args, source, target, edge_probability)

        out_dir = work_dir / "out"
        started = perf_counter()
        completed = subprocess.run(
            [str(fred_binary), "-p", str(model_path), "-r", str(args.replicate), "-d", str(out_dir)],
            env=os.environ | {"FRED_HOME": fred_home},
            text=True, capture_output=True,
        )
        elapsed = perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(f"FRED exited {completed.returncode}: {completed.stderr[-4000:]}")

        report = out_dir / f"RUN{args.replicate}" / "BENCH.csv"
        peak_hospitalized = 0
        final = None
        with report.open(newline="") as stream:
            for row in csv.DictReader(stream):
                peak_hospitalized = max(peak_hospitalized, int(row["BENCH.H"]))
                final = row
        if final is None:
            raise RuntimeError("FRED produced an empty BENCH.csv")

    return {
        "simulate_seconds": elapsed,
        "final_susceptible": int(final["BENCH.S"]),
        "final_exposed": int(final["BENCH.E"]),
        "final_infected": int(final["BENCH.I"]),
        "final_hospitalized": int(final["BENCH.H"]),
        "final_recovered": int(final["BENCH.R"]),
        "peak_hospitalized": peak_hospitalized,
    }


if __name__ == "__main__":
    main("FRED", run_fred, preload=())
