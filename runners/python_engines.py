#!/usr/bin/env python3
"""Single-replicate runners for the Python epidemic engines."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("covasim", "EoN", "epydemic"), required=True)
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


def discrete_transmission_probability(target_r0: float, degree: float, recovery: float) -> float:
    """Per-day edge probability matching R0 via geometric competing risks."""
    transmissibility = min(0.999, target_r0 / max(1.0, degree - 1.0))
    return transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery))


def hospitalization_daily_probability(lifetime_probability: float, recovery: float) -> float:
    return lifetime_probability * recovery / (
        1.0 - lifetime_probability * (1.0 - recovery)
    )


def make_graph(n: int, source: np.ndarray, target: np.ndarray):
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(source.tolist(), target.tolist()))
    return graph


def run_eon(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import EoN
    import networkx as nx

    graph = make_graph(args.n, source, target)
    gamma = 1.0 / args.infectious_days
    sigma = 1.0 / args.latent_days
    tau = (args.target_r0 / max(1.0, args.mean_degree - 1.0)) * gamma
    tau /= 1.0 - args.target_r0 / max(1.0, args.mean_degree - 1.0)
    hospital_rate = args.hospitalization_probability / (1.0 - args.hospitalization_probability) * gamma

    spontaneous = nx.DiGraph()
    spontaneous.add_edge("E", "I", rate=sigma)
    spontaneous.add_edge("I", "H", rate=hospital_rate)
    spontaneous.add_edge("I", "R", rate=gamma)
    spontaneous.add_edge("H", "R", rate=1.0 / args.hospital_days)
    induced = nx.DiGraph()
    induced.add_edge(("I", "S"), ("I", "E"), rate=tau)

    initial = {node: "S" for node in range(args.n)}
    chooser = random.Random(args.seed)
    for node in chooser.sample(range(args.n), min(args.initial_infected, args.n)):
        initial[node] = "I"
    random.seed(args.seed)
    np.random.seed(args.seed)
    started = perf_counter()
    times, susceptible, exposed, infected, hospitalized, recovered = EoN.Gillespie_simple_contagion(
        graph,
        spontaneous,
        induced,
        initial,
        return_statuses=("S", "E", "I", "H", "R"),
        tmax=args.days,
        return_full_data=False,
    )
    elapsed = perf_counter() - started
    return {
        "simulate_seconds": elapsed,
        "final_susceptible": int(susceptible[-1]),
        "final_exposed": int(exposed[-1]),
        "final_infected": int(infected[-1]),
        "final_hospitalized": int(hospitalized[-1]),
        "final_recovered": int(recovered[-1]),
        "peak_hospitalized": int(np.max(hospitalized)),
    }


def run_epydemic(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import epydemic
    from epydemic import CompartmentedModel

    class SEIRH(CompartmentedModel):
        S, E, I, H, R = "S", "E", "I", "H", "R"
        SI = "SEIRH.SI"

        def __init__(self) -> None:
            super().__init__()
            self.peak_hospitalized = 0

        def build(self, params):
            super().build(params)
            initial_fraction = min(1.0, args.initial_infected / args.n)
            self.addCompartment(self.S, 1.0 - initial_fraction)
            self.addCompartment(self.E, 0.0)
            self.addCompartment(self.I, initial_fraction)
            self.addCompartment(self.H, 0.0)
            self.addCompartment(self.R, 0.0)
            self.trackEdgesBetweenCompartments(self.S, self.I, name=self.SI)
            self.trackNodesInCompartment(self.E)
            self.trackNodesInCompartment(self.I)
            self.trackNodesInCompartment(self.H)
            recovery = 1.0 / args.infectious_days
            beta = discrete_transmission_probability(args.target_r0, args.mean_degree, recovery)
            hosp = hospitalization_daily_probability(args.hospitalization_probability, recovery)
            self.addEventPerElement(self.SI, beta, self.infect)
            self.addEventPerElement(self.E, 1.0 / args.latent_days, self.progress)
            # Hospitalization is registered first; locus membership prevents a
            # second transition if both competing events were sampled.
            self.addEventPerElement(self.I, hosp, self.hospitalize)
            self.addEventPerElement(self.I, recovery, self.recover)
            self.addEventPerElement(self.H, 1.0 / args.hospital_days, self.recover)

        def initialCompartments(self):
            infected = set(
                epydemic.rng.choice(
                    args.n, size=min(args.initial_infected, args.n), replace=False
                ).tolist()
            )
            for node in self.network().nodes():
                self.changeInitialCompartment(node, self.I if node in infected else self.S)

        def infect(self, _time, edge):
            susceptible, _infected = edge
            self.changeCompartment(susceptible, self.E)

        def progress(self, _time, node):
            self.changeCompartment(node, self.I)

        def hospitalize(self, _time, node):
            self.changeCompartment(node, self.H)
            self.peak_hospitalized = max(self.peak_hospitalized, len(self.locus(self.H)))

        def recover(self, _time, node):
            self.changeCompartment(node, self.R)

        def results(self):
            result = super().results()
            result["peak_hospitalized"] = self.peak_hospitalized
            return result

    graph = make_graph(args.n, source, target)
    process = SEIRH()
    process.setMaximumTime(float(args.days) + 1.0)
    dynamics = epydemic.SynchronousDynamics(process, graph)
    epydemic.rng.bit_generator.state = np.random.PCG64(args.seed).state
    started = perf_counter()
    result = dynamics.set({}).run(fatal=True)["results"]
    elapsed = perf_counter() - started
    return {
        "simulate_seconds": elapsed,
        "final_susceptible": int(result[SEIRH.S]),
        "final_exposed": int(result[SEIRH.E]),
        "final_infected": int(result[SEIRH.I]),
        "final_hospitalized": int(result[SEIRH.H]),
        "final_recovered": int(result[SEIRH.R]),
        "peak_hospitalized": int(result["peak_hospitalized"]),
    }


def run_covasim(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import covasim as cv

    recovery = 1.0 / args.infectious_days
    beta = discrete_transmission_probability(args.target_r0, args.mean_degree, recovery)
    fixed = lambda value: {"dist": "normal_int", "par1": value, "par2": 0.0}
    prognoses = {
        "age_cutoffs": np.array([0]),
        "sus_ORs": np.array([1.0]),
        "trans_ORs": np.array([1.0]),
        "symp_probs": np.array([1.0]),
        "comorbidities": np.array([1.0]),
        "severe_probs": np.array([args.hospitalization_probability]),
        "crit_probs": np.array([0.0]),
        "death_probs": np.array([0.0]),
    }
    durations = {
        "exp2inf": fixed(args.latent_days),
        "inf2sym": fixed(0),
        "sym2sev": fixed(args.infectious_days),
        "sev2crit": fixed(args.hospital_days),
        "asym2rec": fixed(args.infectious_days),
        "mild2rec": fixed(args.infectious_days),
        "sev2rec": fixed(args.hospital_days),
        "crit2rec": fixed(args.hospital_days),
        "crit2die": fixed(args.hospital_days),
    }
    popdict = {
        "uid": np.arange(args.n, dtype=np.int64),
        "age": np.full(args.n, 40.0),
        "sex": np.zeros(args.n, dtype=np.int64),
        "contacts": {
            "benchmark": {
                "p1": source.astype(np.int64, copy=False),
                "p2": target.astype(np.int64, copy=False),
            }
        },
        "layer_keys": ["benchmark"],
    }
    pars = {
        "pop_size": args.n,
        "n_days": args.days,
        "pop_infected": min(args.initial_infected, args.n),
        "rand_seed": args.seed,
        "verbose": 0,
        "beta": beta,
        "contacts": {"benchmark": args.mean_degree},
        "beta_layer": {"benchmark": 1.0},
        "dynam_layer": {"benchmark": 0},
        "iso_factor": {"benchmark": 1.0},
        "quar_factor": {"benchmark": 1.0},
        "dur": durations,
        "prog_by_age": False,
        "prognoses": prognoses,
        # Restrict Covasim to the common SEIRH target: recovered people remain
        # removed, and transmission has neither person-level dispersion nor a
        # time-varying viral-load multiplier. Covasim still uses its native
        # symptom/severe state bookkeeping to represent hospitalization.
        "use_waning": False,
        "beta_dist": {"dist": "normal", "par1": 1.0, "par2": 0.0},
        "viral_dist": {"frac_time": 1.0, "load_ratio": 1.0, "high_cap": 1_000_000.0},
        "rescale": False,
    }
    sim = cv.Sim(pars=pars, people=popdict)
    started = perf_counter()
    sim.run()
    elapsed = perf_counter() - started
    results = sim.results
    people = sim.people
    recovered = int(np.count_nonzero(people.recovered))
    hospitalized = int(np.count_nonzero(people.severe & ~people.recovered))
    infected = int(np.count_nonzero(people.infectious & ~people.severe & ~people.recovered))
    exposed = int(np.count_nonzero(people.exposed & ~people.infectious & ~people.recovered))
    susceptible = args.n - recovered - hospitalized - infected - exposed
    return {
        "simulate_seconds": elapsed,
        "final_susceptible": susceptible,
        "final_exposed": exposed,
        "final_infected": infected,
        "final_hospitalized": hospitalized,
        "final_recovered": recovered,
        "peak_hospitalized": int(np.max(results["n_severe"].values)),
    }


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


def main() -> None:
    args = parse_args()
    # Match the R runner's timing boundary: interpreter and engine imports are
    # process startup costs and are excluded from setup/simulation timings.
    __import__(args.engine)
    if args.engine in {"EoN", "epydemic"}:
        __import__("networkx")
    total_started = perf_counter()
    source, target = read_edges(args.network)
    if len(source) != args.network_edges:
        raise ValueError(f"edge count mismatch: expected {args.network_edges}, read {len(source)}")
    runners = {"covasim": run_covasim, "EoN": run_eon, "epydemic": run_epydemic}
    result = runners[args.engine](args, source, target)
    total_elapsed = perf_counter() - total_started
    setup_elapsed = total_elapsed - result["simulate_seconds"]
    record = {
        "status": "ok",
        "engine": args.engine,
        "engine_version": args.engine_version,
        "n": args.n,
        "days": args.days,
        "replicate": args.replicate,
        "seed": args.seed,
        "network_sha256": args.network_sha256,
        "network_edges": args.network_edges,
        "mean_degree": args.mean_degree,
        "target_r0": args.target_r0,
        "transmission_multiplier": 1.0,
        "setup_seconds": setup_elapsed,
        "total_seconds": total_elapsed,
        "fingerprint": args.fingerprint,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **result,
    }
    if sum(record[f"final_{state}"] for state in ("susceptible", "exposed", "infected", "hospitalized", "recovered")) != args.n:
        raise RuntimeError("final compartment counts do not sum to population size")
    atomic_write(args.output, record)


if __name__ == "__main__":
    main()
