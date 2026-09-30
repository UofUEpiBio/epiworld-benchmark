#!/usr/bin/env python3
"""Single-replicate Starsim runner for benchmark scenario 01."""

from __future__ import annotations

import argparse

import numpy as np

from runner_common import main, start_simulate, stop_simulate


def discrete_transmission_probability(target_r0: float, degree: float, recovery: float) -> float:
    """Per-day edge probability matching R0 via geometric competing risks."""
    transmissibility = min(0.999, target_r0 / max(1.0, degree - 1.0))
    return transmissibility * recovery / (1.0 - transmissibility * (1.0 - recovery))


def seirh_classes():
    """Starsim's SEIR plus a hospital stay, and a network read from an edge list.

    Defined in a function so that importing starsim stays out of the timings,
    as for the other Python engines.
    """
    import starsim as ss

    class EdgeListNet(ss.Network):
        """A fixed contact network: its edges never change."""

        def step(self):
            pass

    class SEIRH(ss.SEIR):
        """ss.SEIR with a hospital stay, instead of direct recovery, for a
        share of the infectious. Hospitalized agents do not transmit."""

        def __init__(self, **kwargs):
            super().__init__()
            self.define_pars(
                p_hosp=ss.bernoulli(p=0.0),
                dur_hosp=ss.expon(scale=ss.days(7)),
            )
            self.update_pars(**kwargs)
            self.define_states(
                ss.BoolState("hospitalized", label="Hospitalized"),
                ss.FloatArr("ti_hospitalized", label="Time of hospitalization"),
                reset=["infected"],
                infected=lambda self: self.exposed | self.infectious | self.hospitalized,
            )

        def set_progression(self, uids):
            """The infectious period ends in hospital for a share p_hosp of
            cases, and in recovery for the rest."""
            p = self.pars
            end_infectious = self.ti_infectious[uids] + p.dur_inf.rvs(uids)
            will_hosp = p.p_hosp.rvs(uids)
            hospitalized, recovering = uids[will_hosp], uids[~will_hosp]
            self.ti_hospitalized[hospitalized] = end_infectious[will_hosp]
            self.ti_recovered[hospitalized] = end_infectious[will_hosp] + p.dur_hosp.rvs(hospitalized)
            self.ti_recovered[recovering] = end_infectious[~will_hosp]

        def clear_infection(self, uids):
            super().clear_infection(uids)
            self.hospitalized[uids] = False

        def step_state(self):
            ti = self.ti
            infectious = (self.exposed & (self.ti_infectious <= ti)).uids
            self.exposed[infectious] = False
            self.infectious[infectious] = True
            hospitalized = (self.infectious & (self.ti_hospitalized <= ti)).uids
            self.infectious[hospitalized] = False
            self.hospitalized[hospitalized] = True
            recovered = ((self.infectious | self.hospitalized) & (self.ti_recovered <= ti)).uids
            self.clear_infection(recovered)
            self.recovered[recovered] = True

    return EdgeListNet, SEIRH


def run_starsim(args: argparse.Namespace, source: np.ndarray, target: np.ndarray) -> dict:
    import starsim as ss

    EdgeListNet, SEIRH = seirh_classes()
    recovery = 1.0 / args.infectious_days
    beta = discrete_transmission_probability(args.target_r0, args.mean_degree, recovery)
    beta *= args.transmission_multiplier
    # A Sim runs once, so every replicate builds a new one: everything from
    # here to the end of the run is timed as simulation, as epiworld's run()
    # re-initializes its model.
    started = start_simulate()
    # Exactly initial_infected seed cases. Starsim's SEIR seeds them Exposed,
    # as the epiworld family does.
    seeds = np.random.default_rng(args.seed).choice(
        args.n, size=min(args.initial_infected, args.n), replace=False
    )
    seed_probability = np.zeros(args.n)
    seed_probability[seeds] = 1.0
    disease = SEIRH(
        beta=ss.probperday(beta),
        init_prev=ss.bernoulli(p=lambda self, sim, uids: seed_probability[uids]),
        dur_exp=ss.expon(scale=ss.days(args.latent_days)),
        dur_inf=ss.expon(scale=ss.days(args.infectious_days)),
        p_hosp=ss.bernoulli(p=args.hospitalization_probability),
        dur_hosp=ss.expon(scale=ss.days(args.hospital_days)),
        p_death=ss.bernoulli(p=0.0),
    )
    # All-or-nothing vaccine: Starsim's simple_vx with leaky=False protects
    # each vaccinee fully with probability vaccine_efficacy. A day-0 campaign
    # delivers it, before any transmission, to round(vaccine_coverage * n)
    # agents drawn when the campaign runs.
    rng = np.random.default_rng(args.seed + 1)
    vaccine = ss.campaign_vx(
        years=[0],
        prob=1.0,
        product=ss.simple_vx(efficacy=args.vaccine_efficacy, leaky=False),
        eligibility=lambda sim: ss.uids(np.sort(rng.choice(
            args.n, size=round(args.vaccine_coverage * args.n), replace=False
        ))),
    )
    network = EdgeListNet(p1=source, p2=target, beta=np.ones(len(source)), name="benchmark")
    # Starsim steps every day from start to stop inclusive; the step on day d
    # produces the epiworld family's day d + 1, so `days` steps end on day
    # days - 1.
    sim = ss.Sim(
        n_agents=args.n,
        networks=network,
        diseases=disease,
        interventions=vaccine,
        start=ss.days(0),
        stop=ss.days(args.days - 1),
        dt=ss.days(1),
        rand_seed=args.seed,
        use_aging=False,
        verbose=0,
    )
    sim.init()
    # simple_vx draws who is protected from NumPy's global generator.
    np.random.seed(args.seed)
    sim.run()
    elapsed = stop_simulate(started)
    disease = sim.diseases.seirh
    vaccine = sim.interventions.campaign_vx
    return {
        "simulate_seconds": elapsed,
        "final_susceptible": int(np.count_nonzero(disease.susceptible)),
        "final_exposed": int(np.count_nonzero(disease.exposed)),
        "final_infected": int(np.count_nonzero(disease.infectious)),
        "final_hospitalized": int(np.count_nonzero(disease.hospitalized)),
        "final_recovered": int(np.count_nonzero(disease.recovered)),
        "peak_hospitalized": int(np.max(disease.results.n_hospitalized[:])),
        "vaccinated": int(np.count_nonzero(vaccine.vaccinated)),
        "vaccine_protected": int(np.count_nonzero(vaccine.vaccinated & (disease.rel_sus == 0))),
    }


if __name__ == "__main__":
    main("starsim", run_starsim, preload=("starsim",))
