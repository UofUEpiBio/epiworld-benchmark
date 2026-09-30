#!/usr/bin/env python3
"""Single-replicate epiworldpy runner for benchmark scenario 02.

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
    started = start_simulate()
    model.run(args.days, args.seed)
    elapsed = stop_simulate(started)
    # Extracting the outputs comes after the simulation and is timed apart.
    extract_started = perf_counter()
    # The four outputs, all recorded by epiworld during the run. The
    # transmission tree includes the seed cases, whose source is -1.
    db = model.get_db()
    tree = db.get_transmissions()
    transitions = db.get_hist_transition_matrix(False)
    matrix = np.zeros((args.days + 1, len(STATES), len(STATES)), dtype=np.int64)
    matrix[
        transitions["dates"], transitions["state_from"]["indexes"], transitions["state_to"]["indexes"]
    ] = transitions["counts"]
    incidence = matrix[:, STATES.index("S"), STATES.index("E")]
    # Rows are (virus, day the case was infected, case, secondary infections);
    # case -1 is the model itself.
    repnum = db.get_reproductive_number()
    repnum = repnum[repnum[:, 2] != -1]
    with np.errstate(invalid="ignore"):
        reproductive_number = np.bincount(repnum[:, 1], weights=repnum[:, 3], minlength=args.days + 1) / np.bincount(
            repnum[:, 1], minlength=args.days + 1
        )
    extract_seconds = perf_counter() - extract_started
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
        "extract_seconds": extract_seconds,
        **output_summary(
            args.days, np.count_nonzero(tree["sources"] != -1), matrix, incidence, reproductive_number
        ),
    }


if __name__ == "__main__":
    main("epiworldpy", run_epiworldpy, preload=("epiworldpy",))
