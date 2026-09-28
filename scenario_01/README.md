# Scenario 01: SEIRH + all-or-nothing vaccination

2026-09-28

- [Model](#model)
  - [Vaccine](#vaccine)
  - [Outputs](#outputs)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
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
  - [What the vaccine costs epiworld and
    ixa](#what-the-vaccine-costs-epiworld-and-ixa)
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
| Workers             | 1                                                     |
| Latest run failures | 0                                                     |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| epiworldpy |  10000 |  100 |                 0.004 |  0.004 |  0.004 |
| epiworldR  |  10000 |  100 |                 0.004 |  0.004 |  0.004 |
| ixa        |  10000 |  100 |                 0.004 |  0.004 |  0.005 |
| epiworld   |  10000 |  100 |                 0.004 |  0.004 |  0.005 |
| individual |  10000 |  100 |                 0.023 |  0.022 |  0.024 |
| EoN        |  10000 |  100 |                 0.036 |  0.034 |  0.039 |
| covasim    |  10000 |  100 |                 0.067 |  0.065 |  0.076 |
| starsim    |  10000 |  100 |                 0.178 |  0.172 |  0.186 |
| epydemic   |  10000 |  100 |                 0.211 |  0.199 |  0.225 |
| epiworldR  | 100000 |  100 |                 0.009 |  0.009 |  0.010 |
| epiworldpy | 100000 |  100 |                 0.009 |  0.009 |  0.010 |
| epiworld   | 100000 |  100 |                 0.012 |  0.010 |  0.015 |
| ixa        | 100000 |  100 |                 0.033 |  0.030 |  0.038 |
| individual | 100000 |  100 |                 0.054 |  0.051 |  0.057 |
| EoN        | 100000 |  100 |                 0.220 |  0.213 |  0.232 |
| covasim    | 100000 |  100 |                 0.375 |  0.349 |  0.427 |
| starsim    | 100000 |  100 |                 0.929 |  0.841 |  1.011 |
| epydemic   | 100000 |  100 |                 1.146 |  1.114 |  1.195 |

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
| epiworld   |  10000 |                    1.04 |   0.94 |   1.24 |
| epiworldpy |  10000 |                    0.95 |   0.84 |   1.09 |
| covasim    |  10000 |                   16.83 |  15.72 |  19.78 |
| starsim    |  10000 |                   44.24 |  42.18 |  47.79 |
| EoN        |  10000 |                    8.93 |   8.02 |   9.89 |
| epydemic   |  10000 |                   52.56 |  47.68 |  57.24 |
| ixa        |  10000 |                    1.06 |   1.01 |   1.16 |
| individual |  10000 |                    5.75 |   5.36 |   6.25 |
| epiworld   | 100000 |                    1.25 |   1.07 |   1.54 |
| epiworldpy | 100000 |                    0.97 |   0.87 |   1.15 |
| covasim    | 100000 |                   40.99 |  35.75 |  45.26 |
| starsim    | 100000 |                   95.29 |  83.19 | 111.62 |
| EoN        | 100000 |                   23.74 |  21.06 |  26.13 |
| epydemic   | 100000 |                  124.63 | 110.06 | 133.45 |
| ixa        | 100000 |                    3.46 |   3.07 |   4.12 |
| individual | 100000 |                    5.67 |   5.00 |   6.31 |

### Time to a first result

The simulation time above covers what an engine has to redo for every
replicate on the same network (see [what is
measured](../README.md#run-time)). *Build* is the rest of the setup:
turning the edge list into the engine’s population and network, which a
user does once per network. *Build + simulate* is the time to a first
result once the edge list is in memory. Reading the edge file is shown
but left out of it, because its cost depends on each language’s file
parsing rather than on the engine.

| Engine     | Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|-------:|---------------:|----------:|-------------:|---------------------:|
| ixa        |  10000 |          0.003 |     0.000 |        0.004 |                0.004 |
| epiworld   |  10000 |          0.005 |     0.001 |        0.004 |                0.006 |
| epiworldpy |  10000 |          0.017 |     0.008 |        0.004 |                0.011 |
| epiworldR  |  10000 |          0.008 |     0.015 |        0.004 |                0.019 |
| individual |  10000 |          0.008 |     0.016 |        0.023 |                0.039 |
| EoN        |  10000 |          0.017 |     0.033 |        0.036 |                0.070 |
| covasim    |  10000 |          0.018 |     0.045 |        0.067 |                0.113 |
| starsim    |  10000 |          0.018 |     0.000 |        0.178 |                0.178 |
| epydemic   |  10000 |          0.017 |     0.017 |        0.211 |                0.229 |
| epiworld   | 100000 |          0.055 |     0.014 |        0.012 |                0.028 |
| ixa        | 100000 |          0.029 |     0.000 |        0.033 |                0.034 |
| epiworldR  | 100000 |          0.073 |     0.032 |        0.009 |                0.041 |
| epiworldpy | 100000 |          0.173 |     0.055 |        0.009 |                0.064 |
| individual | 100000 |          0.072 |     0.030 |        0.054 |                0.084 |
| covasim    | 100000 |          0.178 |     0.049 |        0.375 |                0.429 |
| EoN        | 100000 |          0.157 |     0.263 |        0.220 |                0.487 |
| starsim    | 100000 |          0.158 |     0.000 |        0.929 |                0.930 |
| epydemic   | 100000 |          0.155 |     0.300 |        1.146 |                1.460 |

### The epiworld family

The wrappers compared with the C++ runner. Here the three draw the
number of protected agents with different random numbers, so they are
matched by seed but do not run identical epidemics. Here the C++ runner
is slower than the wrappers at 100,000 agents, although it runs the same
library. This comes from memory allocation, not from the language layer.
Placing the vaccine allocates about 24,000 tool objects inside `run()`.
The C++ runner starts with a small heap, which grows during those
allocations; R and Python have already grown a large heap while starting
up. Run alone, the C++ runner takes a median 10.9 ms here, and about 9
ms when glibc is told to keep a large heap in reserve
(`MALLOC_TOP_PAD_`) or to serve large blocks from the heap rather than
with `mmap` (`MALLOC_MMAP_THRESHOLD_`). The runner is timed as a C++
user would write it, so the published numbers keep this effect.

| Engine     | Agents | Median time / epiworld |   Q1 |   Q3 |
|:-----------|-------:|-----------------------:|-----:|-----:|
| epiworldR  |  10000 |                   0.96 | 0.81 | 1.07 |
| epiworldpy |  10000 |                   0.87 | 0.75 | 1.02 |
| epiworldR  | 100000 |                   0.80 | 0.65 | 0.94 |
| epiworldpy | 100000 |                   0.80 | 0.63 | 0.93 |

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
| epydemic | 10000 | 0.509 | 0.211 | 0.41 | 0.38 | 0.45 |
| epiworldR | 10000 | 0.009 | 0.004 | 0.44 | 0.44 | 0.50 |
| epiworldpy | 10000 | 0.008 | 0.004 | 0.47 | 0.42 | 0.50 |
| EoN | 10000 | 0.079 | 0.036 | 0.47 | 0.42 | 0.51 |
| epiworld | 10000 | 0.009 | 0.004 | 0.49 | 0.44 | 0.56 |
| ixa | 10000 | 0.007 | 0.004 | 0.60 | 0.57 | 0.65 |
| individual | 10000 | 0.026 | 0.023 | 0.89 | 0.84 | 0.96 |
| starsim | 10000 | 0.174 | 0.178 | 1.02 | 0.96 | 1.08 |
| covasim | 10000 | 0.065 | 0.067 | 1.04 | 0.98 | 1.15 |
| epydemic | 100000 | 1.854 | 1.146 | 0.62 | 0.58 | 0.67 |
| epiworldR | 100000 | 0.015 | 0.009 | 0.67 | 0.58 | 0.77 |
| epiworldpy | 100000 | 0.014 | 0.009 | 0.69 | 0.60 | 0.76 |
| EoN | 100000 | 0.306 | 0.220 | 0.72 | 0.67 | 0.77 |
| epiworld | 100000 | 0.015 | 0.012 | 0.80 | 0.68 | 0.99 |
| ixa | 100000 | 0.038 | 0.033 | 0.84 | 0.74 | 0.99 |
| individual | 100000 | 0.060 | 0.054 | 0.90 | 0.74 | 0.98 |
| starsim | 100000 | 0.938 | 0.929 | 0.98 | 0.89 | 1.06 |
| covasim | 100000 | 0.329 | 0.375 | 1.13 | 1.06 | 1.24 |

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

### What the vaccine costs epiworld and ixa

Both place the vaccine inside the timed call, before the first day.
epiworld clones the tool onto the heap for each of the roughly 24,000
protected agents at 100,000 agents, which takes about 3.2 ms per run;
ixa sets a property on each of the 30,000 vaccinees in a plan at time 0,
about 1.7 ms. Because the vaccine also shrinks the outbreak, this fixed
cost is a larger share of epiworld’s time here than in scenario 00.
ixa’s time per replicate is still mostly rebuilding its context.
[analysis.md](../analysis.md) has the measurements, and explains the
rest of the difference between the two engines.

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
