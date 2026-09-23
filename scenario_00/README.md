# Scenario 00: SEIRH baseline

The reference model for the benchmark. Every later scenario adds complexity to
it, and the report compares each scenario's run time and model code against
this one.

## Model

Susceptible $\rightarrow$ exposed $\rightarrow$ infectious $\rightarrow$
recovered, with a competing infectious $\rightarrow$ hospitalized
$\rightarrow$ recovered branch. Transmission happens along the shared
Watts–Strogatz contact network. There are no interventions.

| Parameter (`scenario.toml`)   | Value | Meaning                                          |
|:------------------------------|------:|:-------------------------------------------------|
| `target_r0`                   |   2.0 | Early-outbreak $R_0$ used by the analytic mapping |
| `initial_infected`            |   100 | Seed cases (10 in the smoke profile)             |
| `latent_days`                 |   4.0 | Mean exposed period                              |
| `infectious_days`             |   7.0 | Mean infectious period                           |
| `hospitalization_probability` |  0.05 | Lifetime probability that an infectious agent is hospitalized |
| `hospital_days`               |   7.0 | Mean hospital stay                               |

Initial cases start infectious in every engine except epiworldR. Its R API
cannot seed a custom model's initial cases in any state other than the one new
infections enter, so they start exposed there.

## Engine notes

- **epiworldR**: a custom model built from `add_state()` and the package's
  update-function factories. New infections enter Exposed.
- **Covasim**: native disease progression restricted to SEIRH. Waning is off,
  and individual transmissibility and viral load are fixed. The severe state
  serves as the hospitalization proxy.
- **EoN**: a continuous-time rate graph run with `fast_simple_contagion`.
- **epydemic**: a `CompartmentedModel` under `SynchronousDynamics`.
- **ixa**: one plan per day, the built-in contact network, and an indexed
  disease-status property. The competing infectious transitions reuse
  epiworldR's roulette rule.

## Calibration

The transmission multipliers in `scenario.toml` align the median day-100
attack rates across engines at each full-profile size. See the report for how
they were chosen.
