# Scenario 05: SEIRH baseline with network uncertainty

2026-10-23

- [Model](#model)
- [Engine notes](#engine-notes)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
  - [Memory](#memory)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
  - [Model lines](#model-lines)
- [Notes](#notes)

[Back to the project overview](../README.md) ·
[Methods](../docs/methods.md) · [Results](../docs/results.md)

This scenario integrates uncertainty in the baseline contact graph:
every replicate draws a separate Watts–Strogatz network, while every
engine uses the same graph for a given replicate.

## Model

The [common model](../docs/methods.md#common-model) with no
interventions, on Watts–Strogatz networks with mean degree 20 and
rewiring probability 0.05, at 10,000 and 100,000 agents. Each engine
runs 100 replicates per size; a deterministic, separately cached graph
is generated for every size/replicate pair and shared across engines for
that pair.

    NULL

## Engine notes

Every engine runs its baseline implementation, described in
[Engines](../docs/methods.md#engines). The three epiworld runners build
the same model and, in this scenario, produce identical epidemics for
every seed.

## Results

> [!WARNING]
>
> No full-profile results are present for this scenario yet. Run
> `make benchmark`, then render this report again.

    NULL

    NULL

### Simulation time

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

    NULL

### Time to a first result

*Build* turns the edge list into the engine’s model, once per network
(and thus once per replicate in this scenario); *build + simulate* is
the time to a first result once the edge list is in memory
([Timing](../docs/methods.md#timing)).

    NULL

### Memory

Median \[Q1, Q3\] in MiB ([Memory](../docs/methods.md#memory)).

    NULL

### Epidemiological sanity checks

    NULL

### Model lines

Counted as described in [Implementation
size](../docs/methods.md#implementation-size), over the regions in
[`code_regions.yml`](code_regions.yml).

| Engine     | Language | Files | Model lines |
|:-----------|:---------|------:|------------:|
| epiworld   | C++      |     1 |          26 |
| individual | R        |     1 |          32 |
| epiworldpy | Python   |     1 |          35 |
| EoN\*      | Python   |     1 |          38 |
| ABM\*      | R        |     1 |          42 |
| epiworldR  | R        |     1 |          47 |
| EpiModel   | R        |     1 |          59 |
| covasim    | Python   |     1 |          65 |
| Agents.jl  | Julia    |     1 |          68 |
| epydemic   | Python   |     1 |          71 |
| starsim    | Python   |     1 |          84 |
| FRED       | Python   |     1 |          92 |
| ixa        | Rust     |     3 |         138 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

## Notes

**Calibration.** All engines initially run with multiplier 1.0. Scenario
00’s factors were estimated on a fixed degree-10 graph and are not valid
for this degree-20, graph-varying design. A later calibration will
estimate factors against this scenario’s 100 network draws before any
calibrated results are published.

**Model lines.** They vary with how much of the model an engine
provides. EoN describes transitions as rate graphs, Covasim overrides
its native parameters to restrict itself to SEIRH, Starsim subclasses
its SEIR for the hospital state, the ixa, individual, and Agents.jl
runners write their daily step or infection process by hand, and the
EpiModel runner writes its infection, progression, and prevalence
modules. The FRED runner’s lines are the Python that writes FRED’s model
file and population.
