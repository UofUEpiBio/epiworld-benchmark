# Scenario 01: SEIRH + all-or-nothing vaccination

2026-09-23

- [Model](#model)
  - [Vaccine](#vaccine)
  - [Outputs](#outputs)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
- [Cost of the added complexity](#cost-of-the-added-complexity)
  - [Run time compared with scenario
    00](#run-time-compared-with-scenario-00)
  - [Code required to define the
    model](#code-required-to-define-the-model)
- [Interpretation](#interpretation)
  - [An implementation choice that mattered: epiworldR’s
    `distribute_tool_to_set()`](#an-implementation-choice-that-mattered-epiworldrs-distribute_tool_to_set)
  - [Calibration](#calibration)
- [Recorded versions](#recorded-versions)

[Back to the project overview](../README.md) · [Scenario 00
report](../scenario_00/README.md)

[Scenario 00](../scenario_00/README.md) plus a vaccine given before
transmission starts. The disease, network, seeds, run length, and
calibration are identical. So for any engine, size, and replicate, the
two scenarios differ only in the vaccine, and this report compares them
replicate by replicate.

## Model

### Vaccine

| Parameter (`scenario.toml`) | Value | Meaning |
|:---|---:|:---|
| `vaccine_coverage` | 0.30 | Share of all agents vaccinated |
| `vaccine_efficacy` | 0.80 | Probability that a vaccinated agent is protected |

- Before the first day, `round(vaccine_coverage * n)` agents are chosen
  uniformly at random and vaccinated.
- The vaccine is **all-or-nothing**. Each vaccinated agent is
  independently protected with probability `vaccine_efficacy`. A
  protected agent can never be infected. An unprotected one behaves
  exactly like an unvaccinated agent. A *leaky* vaccine would instead
  lower everyone’s per-contact risk by 80%.
- Vaccination is independent of the initial infections. A seed case that
  also gets vaccinated stays infected.

About 24% of agents end up protected, so the effective reproduction
number falls from about 2 to about 1.5. The disease parameters are those
of scenario 00.

### Outputs

Runners add two fields to each result record:

- `vaccinated`: the number of agents vaccinated.
- `vaccine_protected`: how many of them are protected, including any who
  were also seed cases.

Protected agents who are never infected are counted in
`final_susceptible`. So the five compartments still sum to `n`, and the
attack rate is still `1 - final_susceptible / n`.

### Engine implementations

Each engine uses its own way of expressing the vaccine, so the
model-lines comparison reflects what a user of that engine would write.

- **epiworldR**: a `tool()` with `susceptibility_reduction = 1`.
  epiworld’s C++ library has an all-or-nothing `ToolVaccine`, but
  epiworldR does not expose it. Unprotected vaccinees need no tool, so
  the runner draws the number of protected agents from a binomial and
  lets epiworld place the tool on that many randomly chosen agents.
- **Covasim**: two day-0 `cv.simple_vaccine` interventions targeted with
  `subtarget`. One gives the protected agents `rel_sus = 0`, the other
  leaves the remaining vaccinees unchanged. A single `simple_vaccine`
  would be leaky. Covasim tracks `people.vaccinated` itself.
- **EoN**: protected agents start in a status `"V"` that has no
  transitions, so no induced transmission rate ever reaches them.
- **epydemic**: a `V` compartment with no events. It sits outside the
  S–I edge locus that drives infection. The vaccine is drawn in
  `initialCompartments`, which runs inside the timed simulation call;
  this adds well under a millisecond.
- **ixa**: a `VaccineStatus` property (`Unvaccinated`, `Protected`,
  `Unprotected`) on `Person`. The daily step skips protected neighbours
  when it looks for susceptible contacts.

## Results

> [!TIP]
>
> The complete 1,000-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 1                                                     |
| Latest run failures | 0                                                     |

| Engine    | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:----------|-------:|-----:|----------------------:|-------:|-------:|
| ixa       |  10000 |  100 |                 0.003 |  0.002 |  0.004 |
| epiworldR |  10000 |  100 |                 0.014 |  0.013 |  0.015 |
| EoN       |  10000 |  100 |                 0.039 |  0.037 |  0.043 |
| covasim   |  10000 |  100 |                 0.067 |  0.064 |  0.071 |
| epydemic  |  10000 |  100 |                 0.361 |  0.255 |  0.414 |
| ixa       | 100000 |  100 |                 0.002 |  0.002 |  0.003 |
| epiworldR | 100000 |  100 |                 0.030 |  0.029 |  0.032 |
| EoN       | 100000 |  100 |                 0.229 |  0.225 |  0.235 |
| covasim   | 100000 |  100 |                 0.359 |  0.355 |  0.364 |
| epydemic  | 100000 |  100 |                 1.168 |  1.156 |  1.189 |

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine   | Agents | Median time / epiworldR |    Q1 |    Q3 |
|:---------|-------:|------------------------:|------:|------:|
| covasim  |  10000 |                    4.98 |  4.48 |  5.35 |
| EoN      |  10000 |                    2.88 |  2.56 |  3.28 |
| epydemic |  10000 |                   26.03 | 18.20 | 30.10 |
| ixa      |  10000 |                    0.23 |  0.16 |  0.31 |
| covasim  | 100000 |                   11.89 | 11.13 | 12.79 |
| EoN      | 100000 |                    7.64 |  7.14 |  8.12 |
| epydemic | 100000 |                   38.99 | 36.27 | 41.29 |
| ixa      | 100000 |                    0.08 |  0.07 |  0.09 |

### Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show the final attack rate, peak hospitalization
load, and the share of agents the vaccine protected.

| Engine | Agents | Median final attack rate | Median peak hospitalized | Median share protected |
|:---|---:|---:|---:|---:|
| covasim | 10000 | 0.105 | 11 | 0.24 |
| EoN | 10000 | 0.113 | 10 | 0.24 |
| epiworldR | 10000 | 0.118 | 8 | 0.24 |
| epydemic | 10000 | 0.121 | 11 | 0.24 |
| ixa | 10000 | 0.119 | 8 | 0.24 |
| covasim | 100000 | 0.012 | 11 | 0.24 |
| EoN | 100000 | 0.013 | 10 | 0.24 |
| epiworldR | 100000 | 0.014 | 9 | 0.24 |
| epydemic | 100000 | 0.014 | 11 | 0.24 |
| ixa | 100000 | 0.014 | 9 | 0.24 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

## Cost of the added complexity

### Run time compared with scenario 00

Each replicate is paired with the scenario 00 replicate that has the
same engine, population size, and seed. Values above one mean that this
scenario took longer.

| Engine | Agents | Median scenario 00 (s) | Median scenario 01 (s) | Median time / scenario 00 | Q1 | Q3 |
|:---|---:|---:|---:|---:|---:|---:|
| EoN | 10000 | 0.083 | 0.039 | 0.48 | 0.43 | 0.54 |
| epiworldR | 10000 | 0.022 | 0.014 | 0.61 | 0.57 | 0.67 |
| epydemic | 10000 | 0.526 | 0.361 | 0.71 | 0.48 | 0.78 |
| ixa | 10000 | 0.004 | 0.003 | 0.74 | 0.48 | 0.95 |
| covasim | 10000 | 0.068 | 0.067 | 0.97 | 0.93 | 1.05 |
| ixa | 100000 | 0.008 | 0.002 | 0.29 | 0.27 | 0.32 |
| epiworldR | 100000 | 0.062 | 0.030 | 0.50 | 0.41 | 0.57 |
| epydemic | 100000 | 1.867 | 1.168 | 0.63 | 0.59 | 0.66 |
| EoN | 100000 | 0.327 | 0.229 | 0.71 | 0.67 | 0.74 |
| covasim | 100000 | 0.369 | 0.359 | 0.97 | 0.95 | 1.01 |

Run time does not isolate the cost of the vaccine machinery. The vaccine
cuts the median attack rate from about 0.39 to 0.12 at 10,000 agents and
from 0.061 to 0.013 at 100,000. Engines whose work tracks the number of
active infections (EoN, epydemic, epiworldR, and ixa) get faster for
that reason alone. Covasim’s vectorized daily update touches every agent
regardless of the outbreak’s size, so its time barely moves. A ratio
below one therefore does not mean the vaccine is free. It means that the
engine’s handling of the vaccine costs less than the smaller outbreak
saves. A ratio above one would mean the feature itself is expensive, as
described below for epiworldR’s `distribute_tool_to_set()`.

### Code required to define the model

A rough measure of effort: how much code each engine needs to express
this scenario, and how much it added to scenario 00. The counting rules
are described in the [project
overview](../README.md#measuring-implementation-effort), and the counted
regions are listed in [`code_regions.yml`](code_regions.yml).

| Engine | Language | Files | Lines, scenario 00 | Lines, scenario 01 | Added since scenario 00 |
|:---|:---|---:|---:|---:|---:|
| EoN | Python | 1 | 37 | 44 | 7 |
| epiworldR | R | 1 | 47 | 62 | 15 |
| covasim | Python | 1 | 64 | 75 | 11 |
| epydemic | Python | 1 | 70 | 88 | 18 |
| ixa | Rust | 3 | 133 | 164 | 31 |

EoN and epydemic only need a compartment with no transitions. Covasim
has a vaccine intervention, but it is leaky, so an all-or-nothing
vaccine takes two targeted doses. epiworldR has a tool that removes
susceptibility. ixa adds an agent property that its hand-written
transmission step has to consult.

## Interpretation

### An implementation choice that mattered: epiworldR’s `distribute_tool_to_set()`

The most direct epiworldR implementation draws the protected agents in R
and hands them to the tool with `distribute_tool_to_set()`. In epiworldR
0.15.1.0 that function is quadratic in the number of agents. Every
per-agent `add_tool()` clones the tool, and the clone carries a copy of
the whole list of target IDs. Placing the tool on about 24,000 named
agents took roughly 4.6 seconds per run at 100,000 agents, against 0.03
seconds with the approach the runner uses now: draw the number of
protected agents, and let epiworld place the tool at random. Both give
the same vaccine statistically. The upstream fix is to share the ID list
between clones rather than copy it, as `distribute_tool_randomly()`
already does.

### Calibration

The transmission multipliers are copied from scenario 00. They correct
each engine’s transmission semantics rather than anything specific to
this scenario, so they were not recalibrated. The table above shows that
the engines’ attack rates stay aligned with the vaccine in place.

## Recorded versions

| Engine    | Recorded version |
|:----------|:-----------------|
| covasim   | 3.1.8            |
| EoN       | 1.92             |
| epiworldR | 0.15.1.0         |
| epydemic  | 1.14.1           |
| ixa       | 3.1.0            |
