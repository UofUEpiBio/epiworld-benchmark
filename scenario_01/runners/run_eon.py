#!/usr/bin/env python3
"""Single-replicate EoN runner for benchmark scenario 01."""

from __future__ import annotations

import argparse
import random
from time import perf_counter

import numpy as np

from runner_common import main


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
    # The initial conditions differ in every run, so building them is timed with
    # the simulation, as epiworld resets its agents, seeds the infections, and
    # distributes its tools inside run().
    started = perf_counter()
    initial = {node: "S" for node in range(args.n)}
    # Protected agents get a status "V" that has no transitions, so no induced
    # rate ever reaches them.
    vaccinated, protected = draw_vaccine(args, rng)
    for node in protected.tolist():
        initial[node] = "V"
    chooser = random.Random(args.seed)
    for node in chooser.sample(range(args.n), min(args.initial_infected, args.n)):
        initial[node] = "I"
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


if __name__ == "__main__":
    main("EoN", run_eon, preload=("EoN", "networkx"))
