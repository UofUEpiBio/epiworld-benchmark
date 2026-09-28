#!/usr/bin/env python3
"""Single-replicate epydemic runner for benchmark scenario 01."""

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


if __name__ == "__main__":
    main("epydemic", run_epydemic, preload=("epydemic", "networkx"))
