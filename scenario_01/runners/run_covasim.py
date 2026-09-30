#!/usr/bin/env python3
"""Single-replicate covasim runner for benchmark scenario 01."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from runner_common import main, start_simulate, stop_simulate


def discrete_transmission_probability(target_r0: float, degree: float, recovery: float) -> float:
    """Per-day edge probability matching R0 via geometric competing risks."""
    transmissibility = min(0.999, target_r0 / max(1.0, degree - 1.0))
    return transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery))


def draw_vaccine(args: argparse.Namespace, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """All-or-nothing vaccine: the vaccinated agents and the protected subset."""
    vaccinated = rng.choice(args.n, size=round(args.vaccine_coverage * args.n), replace=False)
    protected = vaccinated[rng.random(len(vaccinated)) < args.vaccine_efficacy]
    return vaccinated, protected


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
    # The draw differs in every run, so it is timed with the simulation, as
    # epiworld distributes its tools inside run(); Covasim applies the doses on
    # day 0 inside sim.run(). Building the intervention objects is not timed,
    # as building epiworld's Tool is not (the first simple_vaccine() in a
    # process also pays a one-off ~50 ms warm-up).
    vaccine_started = perf_counter()
    vaccinated, protected = draw_vaccine(args, np.random.default_rng(args.seed))
    unprotected = np.setdiff1d(vaccinated, protected)
    vaccine_seconds = perf_counter() - vaccine_started
    everyone_in = lambda inds: {"inds": inds, "vals": 1.0}
    pars["interventions"] = [
        cv.simple_vaccine(days=0, prob=0.0, rel_sus=0.0, rel_symp=1.0, subtarget=everyone_in(protected)),
        cv.simple_vaccine(days=0, prob=0.0, rel_sus=1.0, rel_symp=1.0, subtarget=everyone_in(unprotected)),
    ]
    sim = cv.Sim(pars=pars, people=popdict)
    started = start_simulate()
    sim.run()
    elapsed = stop_simulate(started)
    results = sim.results
    people = sim.people
    recovered = int(np.count_nonzero(people.recovered))
    hospitalized = int(np.count_nonzero(people.severe & ~people.recovered))
    infected = int(np.count_nonzero(people.infectious & ~people.severe & ~people.recovered))
    exposed = int(np.count_nonzero(people.exposed & ~people.infectious & ~people.recovered))
    susceptible = args.n - recovered - hospitalized - infected - exposed
    return {
        "simulate_seconds": elapsed + vaccine_seconds,
        "final_susceptible": susceptible,
        "final_exposed": exposed,
        "final_infected": infected,
        "final_hospitalized": hospitalized,
        "final_recovered": recovered,
        "peak_hospitalized": int(np.max(results["n_severe"].values)),
        "vaccinated": int(np.count_nonzero(people.vaccinated)),
        "vaccine_protected": int(np.count_nonzero(people.rel_sus == 0)),
    }


if __name__ == "__main__":
    main("covasim", run_covasim, preload=("covasim",))
