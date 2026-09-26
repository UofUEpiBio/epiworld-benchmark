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
    parser.add_argument("--engine", choices=("covasim", "EoN", "epydemic", "epiworldpy"), required=True)
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


def discrete_transmission_probability(target_r0: float, degree: float, recovery: float) -> float:
    """Per-day edge probability matching R0 via geometric competing risks."""
    transmissibility = min(0.999, target_r0 / max(1.0, degree - 1.0))
    return transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery))


def hospitalization_daily_probability(lifetime_probability: float, recovery: float) -> float:
    return lifetime_probability * recovery / (
        1.0 - lifetime_probability * (1.0 - recovery)
    )


def draw_vaccine(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """All-or-nothing vaccine: the vaccinated agents and the protected subset."""
    vaccinated = rng.choice(args.n, size=round(args.vaccine_coverage * args.n), replace=False)
    protected = vaccinated[rng.random(len(vaccinated)) < args.vaccine_efficacy]
    return vaccinated, protected


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

    # EoN 1.92 draws from a private Generator, so seeding random/np.random
    # globally never reaches it; the generator has to be passed in explicitly.
    rng = np.random.default_rng(args.seed)
    # Protected agents get a status "V" that has no transitions, so no induced
    # rate ever reaches them.
    vaccinated, protected = draw_vaccine(args, rng)
    initial = {node: "S" for node in range(args.n)}
    for node in protected.tolist():
        initial[node] = "V"
    chooser = random.Random(args.seed)
    for node in chooser.sample(range(args.n), min(args.initial_infected, args.n)):
        initial[node] = "I"
    started = perf_counter()
    # fast_simple_contagion is the event-driven counterpart of
    # Gillespie_simple_contagion: same continuous-time Markov chain, same
    # arguments, and the algorithm EoN's own documentation recommends.
    times, susceptible, exposed, infected, hospitalized, recovered, immune = EoN.fast_simple_contagion(
        graph,
        spontaneous,
        induced,
        initial,
        return_statuses=("S", "E", "I", "H", "R", "V"),
        tmax=args.days,
        rng=rng,
        return_full_data=False,
    )
    elapsed = perf_counter() - started
    return {
        "simulate_seconds": elapsed,
        # Protected agents who were never infected count as susceptible.
        "final_susceptible": int(susceptible[-1] + immune[-1]),
        "final_exposed": int(exposed[-1]),
        "final_infected": int(infected[-1]),
        "final_hospitalized": int(hospitalized[-1]),
        "final_recovered": int(recovered[-1]),
        "peak_hospitalized": int(np.max(hospitalized)),
        "vaccinated": len(vaccinated),
        "vaccine_protected": len(protected),
    }


def run_epydemic(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import epydemic
    from epydemic import CompartmentedModel

    class SEIRHV(CompartmentedModel):
        S, E, I, H, R, V = "S", "E", "I", "H", "R", "V"
        SI = "SEIRHV.SI"

        def __init__(self) -> None:
            super().__init__()
            self.peak_hospitalized = 0
            self.vaccinated = 0
            self.protected = 0

        def build(self, params):
            super().build(params)
            initial_fraction = min(1.0, args.initial_infected / args.n)
            self.addCompartment(self.S, 1.0 - initial_fraction)
            self.addCompartment(self.E, 0.0)
            self.addCompartment(self.I, initial_fraction)
            self.addCompartment(self.H, 0.0)
            self.addCompartment(self.R, 0.0)
            # Protected agents sit in V, which has no events and is outside
            # the S-I edge locus, so transmission never reaches them.
            self.addCompartment(self.V, 0.0)
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
            vaccinated, protected = draw_vaccine(args, epydemic.rng)
            self.vaccinated, self.protected = len(vaccinated), len(protected)
            protected = set(protected.tolist())
            infected = set(
                epydemic.rng.choice(
                    args.n, size=min(args.initial_infected, args.n), replace=False
                ).tolist()
            )
            for node in self.network().nodes():
                if node in infected:
                    compartment = self.I
                elif node in protected:
                    compartment = self.V
                else:
                    compartment = self.S
                self.changeInitialCompartment(node, compartment)

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
            result["vaccinated"] = self.vaccinated
            result["vaccine_protected"] = self.protected
            return result

    graph = make_graph(args.n, source, target)
    process = SEIRHV()
    process.setMaximumTime(float(args.days) + 1.0)
    dynamics = epydemic.SynchronousDynamics(process, graph)
    epydemic.rng.bit_generator.state = np.random.PCG64(args.seed).state
    started = perf_counter()
    result = dynamics.set({}).run(fatal=True)["results"]
    elapsed = perf_counter() - started
    return {
        "simulate_seconds": elapsed,
        # Protected agents who were never infected count as susceptible.
        "final_susceptible": int(result[SEIRHV.S] + result[SEIRHV.V]),
        "final_exposed": int(result[SEIRHV.E]),
        "final_infected": int(result[SEIRHV.I]),
        "final_hospitalized": int(result[SEIRHV.H]),
        "final_recovered": int(result[SEIRHV.R]),
        "peak_hospitalized": int(result["peak_hospitalized"]),
        "vaccinated": int(result["vaccinated"]),
        "vaccine_protected": int(result["vaccine_protected"]),
    }


def run_epiworldpy(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import epiworldpy as epiworld

    recovery = 1.0 / args.infectious_days
    beta = discrete_transmission_probability(args.target_r0, args.mean_degree, recovery)
    beta *= args.transmission_multiplier
    hosp = hospitalization_daily_probability(args.hospitalization_probability, recovery)

    model = epiworld.Model()
    model.add_param(beta, "Transmission rate")
    model.add_param(1.0 / args.latent_days, "Incubation rate")
    model.add_param(hosp, "Hospitalization rate")
    model.add_param(recovery, "Recovery rate")
    model.add_param(1.0 / args.hospital_days, "Hospital recovery rate")
    rate = epiworld.UpdateFun.rate
    # Only infected agents transmit: exposed, hospitalized, and recovered
    # neighbours (states 1, 3, 4) are excluded.
    model.add_state("Susceptible", epiworld.UpdateFun.susceptible(exclude=[1, 3, 4]))
    model.add_state("Exposed", rate(["Incubation rate"], [2]))
    model.add_state("Infected", rate(["Hospitalization rate", "Recovery rate"], [3, 4]))
    model.add_state("Hospitalized", rate(["Hospital recovery rate"], [4]))
    model.add_state("Recovered")

    # New infections enter Exposed (state 1). The initial cases start there
    # too, as in the epiworldR runner.
    pathogen = epiworld.Virus(
        "Benchmark pathogen", min(args.initial_infected, args.n), False, beta, recovery, 0.0
    )
    pathogen.set_state(1, 4, 4)
    pathogen.set_prob_infecting("Transmission rate")
    model.add_virus(pathogen)

    # All-or-nothing vaccine, drawn as in the epiworldR runner: unprotected
    # vaccinees behave exactly like unvaccinated agents, so only the protected
    # ones get a tool, which epiworld places on that many agents at random.
    vaccinated = round(args.vaccine_coverage * args.n)
    protected = int(np.random.default_rng(args.seed).binomial(vaccinated, args.vaccine_efficacy))
    vaccine = epiworld.Tool("Vaccine", protected, False, susceptibility_reduction=1.0)
    model.add_tool(vaccine)
    model.agents_from_edgelist(source.tolist(), target.tolist(), args.n, False)
    model.verbose_off()
    started = perf_counter()
    model.run(args.days, args.seed)
    elapsed = perf_counter() - started
    today = model.get_db().get_today_total()
    final = dict(zip(today["states"]["values"][today["states"]["indexes"]], today["counts"].tolist()))
    history = model.get_db().get_hist_total()
    hist_states = history["states"]["values"][history["states"]["indexes"]]
    return {
        "simulate_seconds": elapsed,
        "final_susceptible": final["Susceptible"],
        "final_exposed": final["Exposed"],
        "final_infected": final["Infected"],
        "final_hospitalized": final["Hospitalized"],
        "final_recovered": final["Recovered"],
        "peak_hospitalized": int(history["counts"][hist_states == "Hospitalized"].max()),
        "vaccinated": vaccinated,
        "vaccine_protected": protected,
    }


def run_covasim(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import covasim as cv

    recovery = 1.0 / args.infectious_days
    beta = discrete_transmission_probability(args.target_r0, args.mean_degree, recovery)
    beta *= args.transmission_multiplier
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
    # All-or-nothing vaccine as two day-0 simple_vaccine doses: the protected
    # subset loses all susceptibility, the rest of the vaccinated are unchanged.
    # simple_vaccine alone is leaky (one rel_sus for everyone it vaccinates).
    vaccinated, protected = draw_vaccine(args, np.random.default_rng(args.seed))
    unprotected = np.setdiff1d(vaccinated, protected)
    everyone_in = lambda inds: {"inds": inds, "vals": 1.0}
    pars["interventions"] = [
        cv.simple_vaccine(days=0, prob=0.0, rel_sus=0.0, rel_symp=1.0, subtarget=everyone_in(protected)),
        cv.simple_vaccine(days=0, prob=0.0, rel_sus=1.0, rel_symp=1.0, subtarget=everyone_in(unprotected)),
    ]
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
        "vaccinated": int(np.count_nonzero(people.vaccinated)),
        "vaccine_protected": int(np.count_nonzero(people.rel_sus == 0)),
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
    runners = {
        "covasim": run_covasim, "EoN": run_eon, "epydemic": run_epydemic,
        "epiworldpy": run_epiworldpy,
    }
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


if __name__ == "__main__":
    main()
