#!/usr/bin/env python3
"""Single-replicate epydemic runner for benchmark scenario 00."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from runner_common import main


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


if __name__ == "__main__":
    main("epydemic", run_epydemic, preload=("epydemic", "networkx"))
