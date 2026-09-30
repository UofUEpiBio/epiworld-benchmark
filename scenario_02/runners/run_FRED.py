#!/usr/bin/env python3
"""Single-replicate FRED runner for benchmark scenario 02.

FRED has no edge-list network primitive comparable to the other engines'
graphs: it is a location-based synthetic-population simulator. This runner
represents the shared network as a synthetic population of single-person
households (so FRED's own household mixing adds no contacts beyond the
network) plus FRED's `Network` group type, whose edges are loaded verbatim
from the benchmark's edge list as `Contact.add_edge` properties.

The all-or-nothing vaccine is a second condition, VAX, written the way FRED's
own `models/vaccine` example writes one: an import picks the vaccinees, each
moves to Protected with probability vaccine_efficacy, and Protected sets the
agent's susceptibility to BENCH to 0.

Scenario 02 extracts four outputs from every run: the transmission tree, daily
incidence, the reproductive number, and the daily transition matrix. FRED's
health records log every exposure, with its source, and every state change;
its daily report counts new exposures.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from runner_common import STATES, main, output_summary, run_measured


def discrete_transmission_probability(target_r0: float, degree: float, recovery: float) -> float:
    """Per-day edge probability matching R0 via geometric competing risks.

    The same mapping as the synchronous Python runners. It applies because
    FRED here reaches each neighbour with probability `transmissibility` per
    day (see write_model) and the infectious period is geometric on 1, 2, ...
    days, as in those engines.
    """
    transmissibility = min(0.999, target_r0 / max(1.0, degree - 1.0))
    return transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery))


def transition_matrix(days: int, day, from_state, to_state) -> np.ndarray:
    """Daily transition counts, indexed [day, from, to] over STATES."""
    matrix = np.zeros((days + 1, len(STATES), len(STATES)), dtype=np.int64)
    np.add.at(matrix, (np.asarray(day, dtype=np.int64), np.asarray(from_state), np.asarray(to_state)), 1)
    return matrix


def case_reproductive_number(days: int, infection_day: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """epiworld's definition: the mean number of secondary infections caused
    by the cases infected on each day. `infection_day` is -1 for agents never
    infected, and `sources` lists the source of every transmission."""
    secondary = np.bincount(sources, minlength=len(infection_day))
    cases = infection_day >= 0
    total = np.bincount(infection_day[cases], weights=secondary[cases], minlength=days + 1)
    count = np.bincount(infection_day[cases], minlength=days + 1)
    with np.errstate(invalid="ignore"):
        return total / count


def read_health_records(path: Path) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The tree as (day, source, target) rows, the transitions as (day, from,
    to) rows, and the seed cases, from FRED's health records."""
    exposure = re.compile(r"day (\d+) person (\d+) age \d+ is EXPOSED to BENCH .* from person (\d+) ")
    seed = re.compile(r"day 0 person (\d+) age \d+ is an IMPORTED EXPOSURE to BENCH$")
    change = re.compile(r"day (\d+) person \d+ .* CONDITION BENCH CHANGES from ([SEIHR]) to ([SEIHR])$")
    tree, transitions, seeds = [], [], []
    index = STATES.index
    with path.open() as stream:
        for line in stream:
            if match := change.search(line):
                transitions.append((int(match[1]), index(match[2]), index(match[3])))
            elif match := exposure.search(line):
                tree.append((int(match[1]), int(match[3]), int(match[2])))
            elif match := seed.search(line):
                seeds.append(int(match[1]))
    return (
        np.asarray(tree, dtype=np.int64).reshape(-1, 3),
        np.asarray(transitions, dtype=np.int64).reshape(-1, 3),
        seeds,
    )


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
    vaccinated = round(args.vaccine_coverage * args.n)
    lines = [
        f"population_directory = {pop_dir}",
        "population_version = bench",
        "country = usa",
        f"locations_file = {locations_file}",
        "start_date = 2020-Jan-01",
        # FRED runs days 0 to days - 1, and day 0 only places the seed cases,
        # which start exposed. One more day gives the other engines' seeding
        # plus `days` days of transmission and progression.
        f"days = {args.days + 1}",
        f"seed = {args.seed}",
        "quality_control = 0",
        # Health records for BENCH in every run (FRED's default is run 1 only).
        "enable_health_records = 1",
        "health_records_run = -1",
        "BENCH.enable_health_records = 1",
        "include_network = Contact",
        "Contact.is_undirected = 1",
        "Contact.can_transmit_BENCH = 1",
        # Without an explicit open-hours schedule, Group_Type::get_time_block
        # returns 0 and the network never transmits. One block covering every
        # hour of every day keeps the network open continuously.
        "Contact.starts_at_hour_0_on_weekdays = 24",
        "Contact.starts_at_hour_0_on_weekends = 24",
        # Network transmission runs once a day with a 24-hour time block and
        # draws about contact_rate * 24 * transmissibility * degree contacts
        # without replacement (Network_Transmission::transmission), so this
        # rate makes transmissibility a per-day, per-neighbour probability.
        "Contact.contact_rate_for_BENCH = 0.0416666666666667",
        "Contact.deterministic_contacts_for_BENCH = 1",
    ]
    lines.extend(f"Contact.add_edge = {a} {b} 1.0" for a, b in zip(source.tolist(), target.tolist()))
    # FRED's geometric(x) is std::geometric_distribution(1/x): failures
    # before the first success, so it starts at 0 and has mean x - 1. One is
    # added so that every stay lasts whole days with mean x, like the
    # synchronous daily engines. Waits are in hours, and every transition
    # lands at hour 0, before that day's transmission.
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
  wait(24*(1+geometric({args.latent_days!r})))
  next(I)
}}
state BENCH.I {{
  set_trans(BENCH,1)
  wait(24*(1+geometric({args.infectious_days!r})))
  next(H) with prob({args.hospitalization_probability!r})
  default(R)
}}
state BENCH.H {{
  set_trans(BENCH,0)
  wait(24*(1+geometric({args.hospital_days!r})))
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
# BENCH is declared first, so its seed cases are imported before VAX draws
# the vaccinees, and a seed case that is also protected stays infected.
condition VAX {{
  states = U V Protected Unprotected Import
  import_start_state = Import
  exposed_state = V
}}
state VAX.U {{
  set_sus(VAX,1)
  wait()
  next()
}}
state VAX.V {{
  set_sus(VAX,0)
  wait(0)
  next(Protected) with prob({args.vaccine_efficacy!r})
  default(Unprotected)
}}
state VAX.Protected {{
  set_sus(BENCH,0)
  wait()
  next()
}}
state VAX.Unprotected {{
  wait()
  next()
}}
# Like BENCH.Import: exactly round(vaccine_coverage * n) agents, drawn
# uniformly from the population, enter V on day 0.
state VAX.Import {{
  import_count({vaccinated})
  wait()
  next()
}}
""")
    model_path = work_dir / "model.fred"
    model_path.write_text("\n".join(lines))
    return model_path


def fred_timings(stdout: str) -> tuple[float, float]:
    """Split one FRED run into reading its input files and everything after.

    FRED prints a lap time for each setup step, measured from process start.
    Up to and including "reading populations", it is parsing the model file
    (with every edge) and the population files: the counterpart of the other
    runners reading the edge list, which is not simulation time. The rest of
    initialization builds the places, population, and network, and FRED must
    redo it for every replicate, so, as for ixa and Starsim, it counts as
    simulation time along with the days themselves.
    """
    laps = re.findall(r"^(.+?) took ([0-9.]+) seconds$", stdout, flags=re.MULTILINE)
    labels = [label for label, _ in laps]
    if "reading populations" not in labels:
        raise RuntimeError("FRED output has no 'reading populations' lap time")
    read = sum(float(seconds) for _, seconds in laps[: labels.index("reading populations") + 1])
    initialization = float(re.search(r"^FRED initialization took ([0-9.]+) seconds$", stdout, re.MULTILINE)[1])
    days = float(re.search(r"Excluding initialization, \d+ days took ([0-9.]+) seconds", stdout)[1])
    return read, initialization - read + days


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
        edge_probability = discrete_transmission_probability(
            args.target_r0, args.mean_degree, 1.0 / args.infectious_days
        ) * args.transmission_multiplier
        model_path = write_model(work_dir, pop_dir, locations_file, args, source, target, edge_probability)

        out_dir = work_dir / "out"
        completed, fred_peak_rss = run_measured(
            [str(fred_binary), "-p", str(model_path), "-r", str(args.replicate), "-d", str(out_dir)],
            env=os.environ | {"FRED_HOME": fred_home},
        )
        if completed.returncode != 0:
            raise RuntimeError(f"FRED exited {completed.returncode}: {completed.stderr[-4000:]}")
        read_seconds, simulate_seconds = fred_timings(completed.stdout)

        report = out_dir / f"RUN{args.replicate}" / "BENCH.csv"
        peak_hospitalized = 0
        rows = []
        with report.open(newline="") as stream:
            for row in csv.DictReader(stream):
                peak_hospitalized = max(peak_hospitalized, int(row["BENCH.H"]))
                rows.append(row)
        if not rows:
            raise RuntimeError("FRED produced an empty BENCH.csv")
        final = rows[-1]
        # Extracting the outputs comes after the simulation and is timed apart.
        extract_started = perf_counter()
        tree, transitions, seeds = read_health_records(out_dir / f"RUN{args.replicate}" / "health_records.txt")
        matrix = transition_matrix(args.days, transitions[:, 0], transitions[:, 1], transitions[:, 2])
        incidence = np.array([int(row["BENCH.newE"]) for row in rows])
        infection_day = np.full(args.n, -1, dtype=np.int64)
        infection_day[seeds] = 0
        infection_day[tree[:, 2]] = tree[:, 0]
        reproductive_number = case_reproductive_number(args.days, infection_day, tree[:, 1])
        extract_seconds = perf_counter() - extract_started
        with (out_dir / f"RUN{args.replicate}" / "VAX.csv").open(newline="") as stream:
            vaccine = list(csv.DictReader(stream))[-1]

    return {
        "engine_read_seconds": read_seconds,
        # FRED runs as a child process, so this runner's own memory says
        # nothing about the engine, and FRED cannot be probed between
        # phases. Its whole-run peak, from wait4(), is the only measure.
        "rss_baseline_bytes": None,
        "rss_after_read_bytes": None,
        "rss_after_setup_bytes": None,
        "peak_rss_setup_bytes": None,
        "peak_rss_simulate_bytes": fred_peak_rss,
        "simulate_seconds": simulate_seconds,
        "final_susceptible": int(final["BENCH.S"]),
        "final_exposed": int(final["BENCH.E"]),
        "final_infected": int(final["BENCH.I"]),
        "final_hospitalized": int(final["BENCH.H"]),
        "final_recovered": int(final["BENCH.R"]),
        "peak_hospitalized": peak_hospitalized,
        "vaccinated": int(vaccine["VAX.Protected"]) + int(vaccine["VAX.Unprotected"]),
        "vaccine_protected": int(vaccine["VAX.Protected"]),
        "extract_seconds": extract_seconds,
        **output_summary(args.days, len(tree), matrix, incidence, reproductive_number),
    }


if __name__ == "__main__":
    main("FRED", run_fred, preload=())
