# Scenario 01: SEIRH + all-or-nothing vaccination

[Scenario 00](../scenario_00/README.md) plus a vaccine given before
transmission starts. The disease, network, seeds, run length, and calibration
are identical, so for any engine, size, and replicate the two scenarios differ
only in the vaccine.

## Vaccine

| Parameter (`scenario.toml`) | Value | Meaning                                         |
|:----------------------------|------:|:------------------------------------------------|
| `vaccine_coverage`          |  0.30 | Share of all agents vaccinated                  |
| `vaccine_efficacy`          |  0.80 | Probability that a vaccinated agent is protected |

- Before the first day, `round(vaccine_coverage * n)` agents are chosen
  uniformly at random and vaccinated.
- The vaccine is **all-or-nothing**. Each vaccinated agent is independently
  protected with probability `vaccine_efficacy`. A protected agent can never
  be infected. An unprotected one behaves exactly like an unvaccinated agent.
  A *leaky* vaccine would instead lower everyone's per-contact risk by 80%.
- Vaccination is independent of the initial infections. A seed case that also
  gets vaccinated stays infected.

About 24% of agents end up protected, so the effective reproduction number
falls from about 2 to about 1.5. Outbreaks are therefore smaller than in
scenario 00. Keep this in mind when comparing run times: part of any change
comes from the extra vaccine bookkeeping, and part comes from a smaller
epidemic.

## Outputs

Runners add two fields to each result record:

- `vaccinated`: the number of agents vaccinated.
- `vaccine_protected`: how many of them are protected, including any who were
  also seed cases.

Protected agents who are never infected are counted in `final_susceptible`.
So the five compartments still sum to `n`, and the attack rate is still
`1 - final_susceptible / n`.

## Engine notes

Each engine uses its own way of expressing the vaccine, so the model-lines
comparison reflects what a user of that engine would write.

- **epiworldR**: a `tool()` with `susceptibility_reduction = 1`. epiworld's
  C++ library has an all-or-nothing `ToolVaccine`, but epiworldR does not
  expose it. Unprotected vaccinees need no tool, so the runner draws the
  number of protected agents from a binomial and lets epiworld place the tool
  on that many randomly chosen agents. `distribute_tool_to_set()` would give
  the same result, but in epiworldR 0.15.1.0 its cost grows quadratically with
  the set size: about 4.6 s for 24,000 agents in the container.
- **Covasim**: two day-0 `cv.simple_vaccine` interventions targeted with
  `subtarget`. One gives the protected agents `rel_sus = 0`, the other leaves
  the remaining vaccinees unchanged. A single `simple_vaccine` would be leaky.
  Covasim tracks `people.vaccinated` itself.
- **EoN**: protected agents start in a status `"V"` that has no transitions,
  so no induced transmission rate ever reaches them.
- **epydemic**: a `V` compartment with no events. It sits outside the S–I
  edge locus that drives infection. The vaccine is drawn in
  `initialCompartments`, which runs inside the timed simulation call.
- **ixa**: a `VaccineStatus` property (`Unvaccinated`, `Protected`,
  `Unprotected`) on `Person`. The daily step skips protected neighbours when
  it looks for susceptible contacts.

## Calibration

The transmission multipliers are copied from scenario 00. They correct each
engine's transmission semantics rather than anything specific to this
scenario, so they were not recalibrated.
