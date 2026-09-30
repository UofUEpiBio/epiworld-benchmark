#!/usr/bin/env python3
"""Single-replicate epydemic runner for benchmark scenario 02.

Scenario 02 extracts four outputs from every run: the transmission tree, daily
incidence, the reproductive number, and the daily transition matrix.
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from runner_common import STATES, main, output_summary, start_simulate, stop_simulate


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
            # The outputs: (day, source, target) for every transmission, and
            # (day, from, to) for every transition.
            self.tree: list[tuple[int, int, int]] = []
            self.transitions: list[tuple[int, int, int]] = []
            self.seeds: list[int] = []

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
            self.seeds = sorted(infected)
            for node in self.network().nodes():
                if node in infected:
                    compartment = self.I
                elif node in protected:
                    compartment = self.V
                else:
                    compartment = self.S
                self.changeInitialCompartment(node, compartment)

        def move(self, time, node, compartment):
            """Change compartment and record the transition. Synchronous
            dynamics runs days at times 1.0, 2.0, and so on."""
            index = STATES.index
            self.transitions.append((int(time), index(self.getCompartment(node)), index(compartment)))
            self.changeCompartment(node, compartment)

        def infect(self, time, edge):
            susceptible, infected = edge
            self.tree.append((int(time), infected, susceptible))
            self.move(time, susceptible, self.E)

        def progress(self, time, node):
            self.move(time, node, self.I)

        def hospitalize(self, time, node):
            self.move(time, node, self.H)
            self.peak_hospitalized = max(self.peak_hospitalized, len(self.locus(self.H)))

        def recover(self, time, node):
            self.move(time, node, self.R)

        def results(self):
            result = super().results()
            result["peak_hospitalized"] = self.peak_hospitalized
            result["vaccinated"] = self.vaccinated
            result["vaccine_protected"] = self.protected
            result["tree"] = np.asarray(self.tree, dtype=np.int64).reshape(-1, 3)
            result["transitions"] = np.asarray(self.transitions, dtype=np.int64).reshape(-1, 3)
            result["seeds"] = self.seeds
            return result

    graph = make_graph(args.n, source, target)
    process = SEIRHV()
    process.setMaximumTime(float(args.days) + 1.0)
    dynamics = epydemic.SynchronousDynamics(process, graph)
    epydemic.rng.bit_generator.state = np.random.PCG64(args.seed).state
    started = start_simulate()
    result = dynamics.set({}).run(fatal=True)["results"]
    elapsed = stop_simulate(started)
    # Extracting the outputs comes after the simulation and is timed apart.
    extract_started = perf_counter()
    tree, transitions = result["tree"], result["transitions"]
    matrix = transition_matrix(args.days, transitions[:, 0], transitions[:, 1], transitions[:, 2])
    incidence = matrix[:, STATES.index("S"), STATES.index("E")]
    infection_day = np.full(args.n, -1, dtype=np.int64)
    infection_day[result["seeds"]] = 0
    infection_day[tree[:, 2]] = tree[:, 0]
    reproductive_number = case_reproductive_number(args.days, infection_day, tree[:, 1])
    extract_seconds = perf_counter() - extract_started
    return {
        "simulate_seconds": elapsed,
        "extract_seconds": extract_seconds,
        # Protected agents who were never infected count as susceptible.
        "final_susceptible": int(result[SEIRHV.S] + result[SEIRHV.V]),
        "final_exposed": int(result[SEIRHV.E]),
        "final_infected": int(result[SEIRHV.I]),
        "final_hospitalized": int(result[SEIRHV.H]),
        "final_recovered": int(result[SEIRHV.R]),
        "peak_hospitalized": int(result["peak_hospitalized"]),
        "vaccinated": int(result["vaccinated"]),
        "vaccine_protected": int(result["vaccine_protected"]),
        **output_summary(args.days, len(tree), matrix, incidence, reproductive_number),
    }


if __name__ == "__main__":
    main("epydemic", run_epydemic, preload=("epydemic", "networkx"))
