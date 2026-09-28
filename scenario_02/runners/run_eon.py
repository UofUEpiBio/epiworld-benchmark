#!/usr/bin/env python3
"""Single-replicate EoN runner for benchmark scenario 02.

Scenario 02 extracts four outputs from every run: the transmission tree, daily
incidence, the reproductive number, and the daily transition matrix.
"""

from __future__ import annotations

import argparse
import math
import random
from time import perf_counter

import numpy as np

from runner_common import STATES, main, output_summary


def draw_vaccine(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """All-or-nothing vaccine: the vaccinated agents and the protected subset."""
    vaccinated = rng.choice(args.n, size=round(args.vaccine_coverage * args.n), replace=False)
    protected = vaccinated[rng.random(len(vaccinated)) < args.vaccine_efficacy]
    return vaccinated, protected


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
    seeds = chooser.sample(range(args.n), min(args.initial_infected, args.n))
    for node in seeds:
        initial[node] = "I"
    # fast_simple_contagion is the event-driven counterpart of
    # Gillespie_simple_contagion: same continuous-time Markov chain, same
    # arguments, and the algorithm EoN's own documentation recommends. With
    # full data it records every node's status history and every transmission.
    sim = EoN.fast_simple_contagion(
        graph,
        spontaneous,
        induced,
        initial,
        return_statuses=("S", "E", "I", "H", "R", "V"),
        tmax=args.days,
        rng=rng,
        return_full_data=True,
    )
    elapsed = perf_counter() - started
    # Extracting the outputs comes after the simulation and is timed apart.
    extract_started = perf_counter()
    # Continuous event times are binned into days: day d covers (d - 1, d].
    tree = np.asarray(sim.transmissions(), dtype=float).reshape(-1, 3)
    tree_day = np.ceil(tree[:, 0]).astype(np.int64)
    tree_source, tree_target = tree[:, 1].astype(np.int64), tree[:, 2].astype(np.int64)
    incidence = np.bincount(tree_day, minlength=args.days + 1)
    infection_day = np.full(args.n, -1, dtype=np.int64)
    infection_day[seeds] = 0
    infection_day[tree_target] = tree_day
    reproductive_number = case_reproductive_number(args.days, infection_day, tree_source)
    index = {status: i for i, status in enumerate(STATES)}
    day, from_state, to_state = [], [], []
    for node in graph:
        node_times, statuses = sim.node_history(node)
        for k in range(1, len(statuses)):
            day.append(math.ceil(node_times[k]))
            from_state.append(index[statuses[k - 1]])
            to_state.append(index[statuses[k]])
    matrix = transition_matrix(args.days, day, from_state, to_state)
    times, counts = sim.summary()
    extract_seconds = perf_counter() - extract_started
    return {
        "simulate_seconds": elapsed,
        "extract_seconds": extract_seconds,
        # Protected agents who were never infected count as susceptible.
        "final_susceptible": int(counts["S"][-1] + counts["V"][-1]),
        "final_exposed": int(counts["E"][-1]),
        "final_infected": int(counts["I"][-1]),
        "final_hospitalized": int(counts["H"][-1]),
        "final_recovered": int(counts["R"][-1]),
        "peak_hospitalized": int(np.max(counts["H"])),
        "vaccinated": len(vaccinated),
        "vaccine_protected": len(protected),
        **output_summary(args.days, len(tree), matrix, incidence, reproductive_number),
    }


if __name__ == "__main__":
    main("EoN", run_eon, preload=("EoN", "networkx"))
