# Scenario 01: SEIRH + all-or-nothing vaccination

2026-09-28

- [Model](#model)
  - [Vaccine](#vaccine)
  - [Outputs](#outputs)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [The epiworld family](#the-epiworld-family)
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
- Distributing the vaccine is timed as part of the simulation in every
  runner, as seeding is (see [scenario 00](../scenario_00/README.md)).
  The draw differs in every run, and epiworld places its tools inside
  `run()`, so the other runners time their draw too. EoN’s timer covers
  building its initial status dictionary. Covasim’s covers the draw, and
  Covasim gives the doses on day 0 inside `sim.run()`. Starsim draws the
  vaccinees when its day-0 campaign runs, inside `sim.run()`. epydemic
  draws inside its simulation call, and ixa in a plan at time 0, inside
  `execute()`. individual’s timer covers its draw. The epiworld runners
  draw only the number of protected agents (one binomial draw) before
  `run()`.

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
  Unprotected vaccinees need no tool, so the runner draws the number of
  protected agents from a binomial and lets epiworld place the tool on
  that many randomly chosen agents.
- **epiworld** (C++) and **epiworldpy**: the same tool, placed the same
  way. Each draws the number of protected agents with its own language’s
  random numbers, so unlike in scenario 00 the three runners’ epidemics
  differ replicate by replicate, though not in distribution. The C++
  library also has an all-or-nothing `ToolVaccine`, which neither
  wrapper exposes. It decides whether an agent is protected only when
  the agent is first exposed, so the number protected, which every
  runner reports, is not observable with it.
- **Covasim**: two day-0 `cv.simple_vaccine` interventions targeted with
  `subtarget`. One gives the protected agents `rel_sus = 0`, the other
  leaves the remaining vaccinees unchanged. A single `simple_vaccine`
  would be leaky. Covasim tracks `people.vaccinated` itself.
- **Starsim**: its own all-or-nothing vaccine,
  `ss.simple_vx(leaky=False)`, delivered by a day-0 `ss.campaign_vx` to
  the drawn agents (the campaign’s eligibility function returns them).
  The vaccine sets `rel_sus = 0` for the protected vaccinees; Starsim
  applies interventions before transmission within a step. `simple_vx`
  draws who is protected from NumPy’s global generator, so the runner
  seeds it.
- **EoN**: protected agents start in a status `"V"` that has no
  transitions, so no induced transmission rate ever reaches them.
- **epydemic**: a `V` compartment with no events. It sits outside the
  S–I edge locus that drives infection. The vaccine is drawn in
  `initialCompartments`, which runs inside the timed simulation call.
- **ixa**: a `VaccineStatus` property (`Unvaccinated`, `Protected`,
  `Unprotected`) on `Person`, set by a plan at time 0. The daily step
  skips protected neighbours when it looks for susceptible contacts.
- **individual**: no intervention objects, so the protected agents are a
  `Bitset`, and the infection process intersects the susceptible agents
  with its complement.

## Results

> [!TIP]
>
> The complete 1,800-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 4                                                     |
| Latest run failures | 0                                                     |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| ixa        |  10000 |  100 |                 0.002 |  0.002 |  0.002 |
| epiworldpy |  10000 |  100 |                 0.005 |  0.004 |  0.005 |
| epiworld   |  10000 |  100 |                 0.005 |  0.004 |  0.005 |
| epiworldR  |  10000 |  100 |                 0.005 |  0.005 |  0.006 |
| individual |  10000 |  100 |                 0.025 |  0.025 |  0.026 |
| EoN        |  10000 |  100 |                 0.043 |  0.040 |  0.049 |
| covasim    |  10000 |  100 |                 0.074 |  0.069 |  0.082 |
| starsim    |  10000 |  100 |                 0.226 |  0.200 |  0.246 |
| epydemic   |  10000 |  100 |                 0.296 |  0.267 |  0.317 |
| ixa        | 100000 |  100 |                 0.004 |  0.004 |  0.005 |
| epiworldpy | 100000 |  100 |                 0.010 |  0.010 |  0.011 |
| epiworldR  | 100000 |  100 |                 0.011 |  0.010 |  0.011 |
| epiworld   | 100000 |  100 |                 0.012 |  0.012 |  0.013 |
| individual | 100000 |  100 |                 0.058 |  0.057 |  0.059 |
| EoN        | 100000 |  100 |                 0.237 |  0.233 |  0.243 |
| covasim    | 100000 |  100 |                 0.369 |  0.365 |  0.376 |
| starsim    | 100000 |  100 |                 1.120 |  1.016 |  1.136 |
| epydemic   | 100000 |  100 |                 1.173 |  1.161 |  1.197 |

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|-------:|------------------------:|-------:|-------:|
| epiworld   |  10000 |                    0.95 |   0.78 |   1.13 |
| epiworldpy |  10000 |                    0.94 |   0.78 |   1.20 |
| covasim    |  10000 |                   14.85 |  13.28 |  17.56 |
| starsim    |  10000 |                   41.51 |  38.37 |  51.28 |
| EoN        |  10000 |                    8.65 |   7.53 |  10.92 |
| epydemic   |  10000 |                   59.08 |  50.20 |  67.29 |
| ixa        |  10000 |                    0.43 |   0.36 |   0.53 |
| individual |  10000 |                    5.00 |   4.33 |   6.00 |
| epiworld   | 100000 |                    1.17 |   1.06 |   1.29 |
| epiworldpy | 100000 |                    0.98 |   0.92 |   1.11 |
| covasim    | 100000 |                   35.12 |  33.20 |  37.31 |
| starsim    | 100000 |                  102.16 |  93.69 | 111.68 |
| EoN        | 100000 |                   22.74 |  20.90 |  23.91 |
| epydemic   | 100000 |                  112.35 | 104.28 | 118.38 |
| ixa        | 100000 |                    0.42 |   0.39 |   0.48 |
| individual | 100000 |                    5.50 |   5.09 |   5.90 |

### The epiworld family

The wrappers compared with the C++ runner. Here the three draw the
number of protected agents with different random numbers, so they are
matched by seed but do not run identical epidemics. As in [scenario
00](../scenario_00/README.md#the-language-layer-costs-nothing-measurable-in-the-simulation),
the C++ runner’s slower simulation at 100,000 agents comes from where
its arrays land in memory, not from the language layer.

| Engine     | Agents | Median time / epiworld |   Q1 |   Q3 |
|:-----------|-------:|-----------------------:|-----:|-----:|
| epiworldR  |  10000 |                   1.05 | 0.88 | 1.28 |
| epiworldpy |  10000 |                   1.00 | 0.85 | 1.16 |
| epiworldR  | 100000 |                   0.85 | 0.78 | 0.94 |
| epiworldpy | 100000 |                   0.85 | 0.79 | 0.92 |

### Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show the final attack rate, peak hospitalization
load, and the share of agents the vaccine protected.

| Engine | Agents | Median final attack rate | Median peak hospitalized | Median share protected |
|:---|---:|---:|---:|---:|
| covasim | 10000 | 0.105 | 11 | 0.24 |
| EoN | 10000 | 0.113 | 10 | 0.24 |
| epiworld | 10000 | 0.119 | 8 | 0.24 |
| epiworldpy | 10000 | 0.117 | 8 | 0.24 |
| epiworldR | 10000 | 0.115 | 9 | 0.24 |
| epydemic | 10000 | 0.121 | 11 | 0.24 |
| individual | 10000 | 0.117 | 9 | 0.24 |
| ixa | 10000 | 0.119 | 8 | 0.24 |
| starsim | 10000 | 0.119 | 9 | 0.24 |
| covasim | 100000 | 0.012 | 11 | 0.24 |
| EoN | 100000 | 0.013 | 10 | 0.24 |
| epiworld | 100000 | 0.014 | 9 | 0.24 |
| epiworldpy | 100000 | 0.014 | 9 | 0.24 |
| epiworldR | 100000 | 0.014 | 9 | 0.24 |
| epydemic | 100000 | 0.014 | 11 | 0.24 |
| individual | 100000 | 0.013 | 10 | 0.24 |
| ixa | 100000 | 0.014 | 9 | 0.24 |
| starsim | 100000 | 0.014 | 10 | 0.24 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

## Cost of the added complexity

### Run time compared with scenario 00

Each replicate is paired with the scenario 00 replicate that has the
same engine, population size, and seed. Values above one mean that this
scenario took longer.

| Engine | Agents | Median scenario 00 (s) | Median scenario 01 (s) | Median time / scenario 00 | Q1 | Q3 |
|:---|---:|---:|---:|---:|---:|---:|
| ixa | 10000 | 0.005 | 0.002 | 0.44 | 0.39 | 0.49 |
| epiworldpy | 10000 | 0.010 | 0.005 | 0.46 | 0.39 | 0.54 |
| epiworld | 10000 | 0.010 | 0.005 | 0.47 | 0.42 | 0.54 |
| epydemic | 10000 | 0.587 | 0.296 | 0.49 | 0.44 | 0.54 |
| epiworldR | 10000 | 0.010 | 0.005 | 0.50 | 0.44 | 0.56 |
| EoN | 10000 | 0.087 | 0.043 | 0.50 | 0.44 | 0.58 |
| covasim | 10000 | 0.097 | 0.074 | 0.77 | 0.67 | 0.86 |
| individual | 10000 | 0.033 | 0.025 | 0.78 | 0.70 | 0.84 |
| starsim | 10000 | 0.178 | 0.226 | 1.24 | 1.12 | 1.38 |
| ixa | 100000 | 0.009 | 0.004 | 0.48 | 0.44 | 0.54 |
| epydemic | 100000 | 1.876 | 1.173 | 0.63 | 0.60 | 0.66 |
| epiworldR | 100000 | 0.016 | 0.011 | 0.65 | 0.59 | 0.71 |
| epiworldpy | 100000 | 0.016 | 0.010 | 0.67 | 0.59 | 0.74 |
| EoN | 100000 | 0.333 | 0.237 | 0.71 | 0.67 | 0.76 |
| epiworld | 100000 | 0.016 | 0.012 | 0.75 | 0.68 | 0.86 |
| individual | 100000 | 0.058 | 0.058 | 0.98 | 0.95 | 1.00 |
| starsim | 100000 | 1.111 | 1.120 | 1.01 | 0.92 | 1.03 |
| covasim | 100000 | 0.361 | 0.369 | 1.02 | 1.01 | 1.04 |

Run time does not isolate the cost of the vaccine machinery. The vaccine
cuts the median attack rate from about 0.39 to 0.12 at 10,000 agents and
from 0.061 to 0.013 at 100,000. Engines whose work tracks the number of
active infections (EoN, epydemic, the epiworld family, and ixa) get
faster for that reason alone. Covasim’s vectorized daily update touches
every agent regardless of the outbreak’s size, and Starsim’s draws
transmission on every edge every day, so their times barely move.
individual’s infection process scans the infectious agents’ neighbours
but also works on full-population bitsets every day, so it sits in
between. A ratio below one therefore does not mean the vaccine is free.
It means that the engine’s handling of the vaccine costs less than the
smaller outbreak saves. A ratio above one would mean the feature itself
is expensive, as epiworldR’s `distribute_tool_to_set()` was before
version 0.16.1 (see below).

### Code required to define the model

A rough measure of effort: how much code each engine needs to express
this scenario, and how much it added to scenario 00. The counting rules
are described in the [project
overview](../README.md#measuring-implementation-effort), and the counted
regions are listed in [`code_regions.yml`](code_regions.yml).

| Engine | Language | Files | Lines, scenario 00 | Lines, scenario 01 | Added since scenario 00 |
|:---|:---|---:|---:|---:|---:|
| epiworld | C++ | 1 | 26 | 34 | 8 |
| individual | R | 1 | 32 | 37 | 5 |
| epiworldpy | Python | 1 | 34 | 38 | 4 |
| EoN | Python | 1 | 37 | 44 | 7 |
| epiworldR | R | 1 | 47 | 62 | 15 |
| covasim | Python | 1 | 64 | 75 | 11 |
| epydemic | Python | 1 | 70 | 88 | 18 |
| starsim | Python | 1 | 83 | 94 | 11 |
| ixa | Rust | 3 | 138 | 168 | 30 |

EoN and epydemic only need a compartment with no transitions. Covasim
has a vaccine intervention, but it is leaky, so an all-or-nothing
vaccine takes two targeted doses. Starsim’s own vaccine product supports
all-or-nothing protection directly, and needs only a campaign to deliver
it. epiworldR has a tool that removes susceptibility. ixa adds an agent
property that its hand-written transmission step has to consult, and
individual a set that its infection process leaves out.

## Interpretation

### An implementation choice that mattered: epiworldR’s `distribute_tool_to_set()`

The most direct epiworldR implementation draws the protected agents in R
and hands them to the tool with `distribute_tool_to_set()`. In epiworldR
0.15.1.0 that function was quadratic in the number of agents. Every
per-agent `add_tool()` cloned the tool, and the clone carried a copy of
the whole list of target IDs. Placing the tool on about 24,000 named
agents took roughly 4.6 seconds per run at 100,000 agents, against 0.03
seconds with the approach the runner uses: draw the number of protected
agents, and let epiworld place the tool at random. Both give the same
vaccine statistically. epiworldR 0.16.1 fixed the quadratic cost, so
either approach is now fast; the runner keeps the second.

### Calibration

The transmission multipliers are copied from scenario 00. They correct
each engine’s transmission semantics rather than anything specific to
this scenario, so they were not recalibrated. The table above shows that
the engines’ attack rates stay aligned with the vaccine in place.

## Recorded versions

| Engine     | Recorded version  |
|:-----------|:------------------|
| covasim    | 3.1.8             |
| EoN        | 1.92              |
| epiworld   | 0.17.0            |
| epiworldpy | 0.17.0-0+g4a1ee0b |
| epiworldR  | 0.17.0.0          |
| epydemic   | 1.14.1            |
| individual | 0.1.19            |
| ixa        | 3.1.0             |
| starsim    | 3.6.1             |
