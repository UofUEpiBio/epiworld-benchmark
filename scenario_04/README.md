# Scenario 04: SEIRH on a collapsed GeoPops network

2026-09-30

- [Model](#model)
  - [The network](#the-network)
  - [Calibration](#calibration)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
  - [The epiworld family](#the-epiworld-family)
  - [Memory](#memory)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
  - [Code required to define the
    model](#code-required-to-define-the-model)
- [Recorded versions](#recorded-versions)

[Back to the project overview](../README.md) · [Scenario 00
report](../scenario_00/README.md)

[Scenario 00](../scenario_00/README.md)’s model on a real contact
network instead of the shared Watts–Strogatz graph: the model, its
parameters, the 100 seed cases, and the runners in [`runners/`](runners)
are copies of scenario 00’s. FRED and Agents.jl were first added in this
scenario and now run in every scenario. This scenario also asks a
different question from scenario 00: does the picture change on a graph
with real degree heterogeneity, rather than Watts–Strogatz’s
near-constant degree?

## Model

See [scenario 00](../scenario_00/README.md#model) for the disease model,
its parameters, and the timing conventions.

### The network

Every earlier scenario shares a synthetic Watts–Strogatz network (mean
degree 10, rewiring probability 0.05, so nearly every agent has close to
the same number of contacts). This scenario instead uses a network built
from
<a href="https://github.com/GeoPopsHub" target="_blank">GeoPops</a>’
<a href="https://github.com/GeoPopsHub/sc_spartanburg_measles"
target="_blank">Spartanburg County, South Carolina synthetic
population</a>: a real, degree-heterogeneous contact structure, at the
cost of not being able to vary population size independently of the
source data.

[`scripts/generate_network.py`](../scripts/generate_network.py)
downloads GeoPops’ four contact-layer adjacency matrices (household,
workplace, school, and group-quarters, each a checksum-verified
`MatrixMarket` file pinned to a commit), keeps the induced subgraph on
the first 200,000 rows of each, and unions the four layers into one
undirected, unweighted graph. Of those 200,000 people, 34,135 (17%) have
no contacts among the others; they could never be infected, so they are
dropped and the remaining 165,865 renumbered in row order. The result
has 376,526 edges and mean degree 4.54. Layer identity, edge weights,
demographic attributes, and contacts touching excluded rows are
discarded; only which pairs of people are ever in contact survives.
[`scenario.toml`](scenario.toml) records the exact commit and per-layer
checksums.

The graph is still fragmented and heterogeneous. It splits into 17,128
components, and the largest holds 127,291 agents (77%), which caps the
attack rate. Degrees range from 1 to 25 (median 3). Because households
form cliques and degrees vary, the shared analytic mapping, which
divides $R_0$ by the mean degree minus one, does not give an $R_0$ of 2
here: the mean excess degree is 6.9, not 3.5. The mapping is the same
for every engine, so the comparison stays aligned, but the outbreaks are
larger than the nominal $R_0$ suggests.

| Parameter (`scenario.toml`) | Value | Meaning |
|:---|---:|:---|
| Population | 165,865 | people with contacts among the first 200,000 GeoPops rows |
| Edges | 376,526 | after union and dedup |
| Mean degree | 4.54 | observed, not fixed |
| Replicates | 20 | fewer than scenario 00; the slowest engines take seconds per replicate here |

Like scenario 03, this scenario runs one replicate at a time
(`workers = 1` in its `[design]` table), so its timings are comparable
to every other scenario’s default sequential runs.

### Calibration

Scenario 00’s per-engine transmission multipliers were derived from full
runs at 10,000 and 100,000 agents on the Watts–Strogatz network, and do
not carry over to a different topology at a different size. Every engine
therefore runs uncalibrated here (`transmission_multiplier` defaults to
1.0) until a full run’s median attack rates give real factors to
calibrate against; see [`scenario.toml`](scenario.toml).

### Engine implementations

The engines from [scenario
00](../scenario_00/README.md#engine-implementations) run the same model
unchanged, reading this scenario’s edge list instead. FRED and
Agents.jl, which scenario 00 also describes, were added here first:

- **FRED** (<a href="https://github.com/PublicHealthDynamicsLab/FRED"
  target="_blank">PublicHealthDynamicsLab/FRED</a>): a compiled,
  DSL-driven simulator with no edge-list network primitive comparable to
  the other engines’ graphs. The runner
  ([`run_FRED.py`](runners/run_FRED.py)) builds a synthetic population
  of 165,865 single-person households (so FRED’s own household mixing
  adds no contacts beyond the network) and loads the benchmark’s edges
  verbatim as `Contact.add_edge` properties on FRED’s `Network` group
  type, with transmission restricted to that network. Each day, an
  infectious agent draws about `transmissibility` × degree contacts
  without replacement among its neighbours, so every neighbour is
  reached with that daily probability, and the runner uses the same
  per-day mapping as the synchronous engines. Latent, infectious, and
  hospital stays are whole days with the configured means. The build
  patches an upstream off-by-one in `Date::setup_dates()` that corrupts
  the heap at this population size (see the
  [Dockerfile](../.devcontainer/Dockerfile)). FRED runs one replicate
  per process and must reread and rebuild everything each time. Its
  timings are split with FRED’s own lap timers: parsing the generated
  model file (which holds every edge) and the population files counts as
  reading, like the other runners’ edge-file parsing; building places,
  the population, and the network, plus the days themselves, count as
  simulation, as ixa’s and Starsim’s per-replicate builds do.
- **Agents.jl** (<a href="https://github.com/JuliaDynamics/Agents.jl"
  target="_blank">JuliaDynamics/Agents.jl</a>): a `StandardABM` with a
  per-agent status and a per-day `model_step!` that scans each
  infectious agent’s adjacency list, matching the synchronous daily
  semantics of epiworldR, epydemic, and individual. The adjacency list
  and transmission math are written for this benchmark, since Agents.jl
  has no built-in epidemic model. Initial cases start infectious, as in
  ixa and individual. Before any timer starts, the runner steps a
  throwaway two-agent model so that Julia compiles its methods;
  otherwise about 0.1 seconds of compilation lands in the first
  simulation.

**MEmilio is deliberately excluded.** Its ABM has no edge-list network
primitive either — agents interact only through shared Locations
(household, work, school, …) — so matching this network would need on
the order of 376,000 synthetic two-person Locations, of unproven
performance, on top of an 8-compartment, viral-load-driven disease model
with no simple SEIRH mapping. See [issue
\#11](https://github.com/UofUEpiBio/epiworld-benchmark/issues/11).

## Results

> [!TIP]
>
> The complete 220-run design for this scenario is available.

| Field | Value |
|:---|:---|
| CPU | Apple M3 Pro (container sees: Apple (model not reported)) |
| Cores | 10 logical, 10 physical |
| Memory | 12.7 GiB |
| OS | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Kernel | 6.12.13-200.fc41.aarch64 |
| Container image | 6ab6f982c445 |
| Python | 3.12.11 |
| R | R version 4.5.1 (2025-06-13) – “Great Square Root” |
| Julia | julia version 1.13.1 |
| Rust | rustc 1.98.0 (88d9e12ae 2026-08-18) |
| C++ compiler | c++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0 |
| Quarto | 1.10.18 |
| Repository commit | 467d650 |
| Workers | 1 |
| Run dates | 2026-09-30 to 2026-09-30 |
| Latest run | 220 executed, 0 cached, 0 failed |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| Agents.jl  | 165865 |   20 |                 0.133 |  0.132 |  0.135 |
| individual | 165865 |   20 |                 0.136 |  0.134 |  0.140 |
| ixa        | 165865 |   20 |                 0.149 |  0.147 |  0.153 |
| epiworldpy | 165865 |   20 |                 0.152 |  0.151 |  0.153 |
| epiworld   | 165865 |   20 |                 0.156 |  0.155 |  0.157 |
| epiworldR  | 165865 |   20 |                 0.157 |  0.155 |  0.158 |
| covasim    | 165865 |   20 |                 0.562 |  0.548 |  0.609 |
| starsim    | 165865 |   20 |                 0.822 |  0.814 |  0.830 |
| FRED       | 165865 |   20 |                 2.658 |  2.579 |  2.710 |
| EoN        | 165865 |   20 |                 3.534 |  3.470 |  3.590 |
| epydemic   | 165865 |   20 |                12.511 | 12.476 | 12.582 |

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by replicate seed. Values above one mean that
epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |    Q1 |    Q3 |
|:-----------|-------:|------------------------:|------:|------:|
| epiworld   | 165865 |                    1.00 |  0.99 |  1.01 |
| epiworldpy | 165865 |                    0.97 |  0.96 |  0.99 |
| covasim    | 165865 |                    3.57 |  3.51 |  3.85 |
| starsim    | 165865 |                    5.26 |  5.22 |  5.37 |
| EoN        | 165865 |                   22.65 | 22.09 | 22.88 |
| epydemic   | 165865 |                   79.84 | 79.29 | 81.50 |
| ixa        | 165865 |                    0.96 |  0.94 |  0.98 |
| individual | 165865 |                    0.87 |  0.86 |  0.89 |
| FRED       | 165865 |                   16.93 | 16.68 | 17.16 |
| Agents.jl  | 165865 |                    0.85 |  0.85 |  0.86 |

### Time to a first result

The simulation time above covers what an engine has to redo for every
replicate on the same network (see [what is
measured](../README.md#run-time)). *Build* is the rest of the setup:
turning the edge list into the engine’s population and network, which a
user does once per network. *Build + simulate* is the time to a first
result once the edge list is in memory.

| Engine     | Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|-------:|---------------:|----------:|-------------:|---------------------:|
| ixa        | 165865 |          0.030 |     0.000 |        0.149 |                0.149 |
| Agents.jl  | 165865 |          0.304 |     0.028 |        0.133 |                0.161 |
| individual | 165865 |          0.064 |     0.034 |        0.136 |                0.169 |
| epiworld   | 165865 |          0.046 |     0.014 |        0.156 |                0.169 |
| epiworldR  | 165865 |          0.064 |     0.032 |        0.157 |                0.188 |
| epiworldpy | 165865 |          0.129 |     0.070 |        0.152 |                0.221 |
| covasim    | 165865 |          0.134 |     0.043 |        0.562 |                0.605 |
| starsim    | 165865 |          0.130 |     0.000 |        0.822 |                0.822 |
| FRED       | 165865 |          2.420 |     0.664 |        2.658 |                3.318 |
| EoN        | 165865 |          0.131 |     0.421 |        3.534 |                3.957 |
| epydemic   | 165865 |          0.127 |     0.398 |       12.511 |               12.905 |

### The epiworld family

| Engine     | Agents | Median time / epiworld |   Q1 |   Q3 |
|:-----------|-------:|-----------------------:|-----:|-----:|
| epiworldR  | 165865 |                   1.00 | 0.99 | 1.01 |
| epiworldpy | 165865 |                   0.97 | 0.97 | 0.98 |

### Memory

Resident memory (RSS) of each run, in MiB, as median \[Q1, Q3\] across
replicates. *Simulation memory* is the peak during the simulate timer
less the memory held when it started: the extra memory one replicate
needs. *Model footprint* is what building the reusable model adds after
the edge list is read, and *baseline* is the runtime after its imports.
FRED runs as a child process, so only its overall peak is known. See
“Memory” under “What is measured” in the
[overview](../README.md#memory).

| Engine | Agents | Baseline (MiB) | Model footprint (MiB) | Simulation memory (MiB) | Overall peak (MiB) |
|:---|:---|---:|---:|---:|---:|
| ixa | 165865 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 37.5 \[37.2, 37.5\] | 45.7 \[45.6, 45.9\] |
| epiworld | 165865 | 3.0 \[3.0, 3.0\] | 29.7 \[29.7, 29.7\] | 54.2 \[54.1, 54.3\] | 90.5 \[90.5, 90.5\] |
| individual | 165865 | 74.8 \[74.8, 74.8\] | 6.1 \[6.1, 6.1\] | 38.9 \[38.7, 39.7\] | 130.4 \[130.2, 131.2\] |
| epiworldpy | 165865 | 46.6 \[46.6, 46.7\] | 41.1 \[40.6, 41.1\] | 41.5 \[41.5, 41.5\] | 136.9 \[136.8, 137.0\] |
| epiworldR | 165865 | 72.1 \[72.1, 72.1\] | 31.3 \[31.3, 31.3\] | 36.2 \[36.2, 36.4\] | 149.6 \[149.6, 149.7\] |
| covasim | 165865 | 225.1 \[225.1, 225.1\] | 3.8 \[3.1, 3.8\] | 70.7 \[70.2, 70.7\] | 307.4 \[306.9, 307.4\] |
| starsim | 165865 | 260.9 \[260.9, 260.9\] | 0.0 \[0.0, 0.0\] | 62.5 \[62.4, 63.3\] | 330.9 \[330.8, 331.7\] |
| EoN | 165865 | 120.3 \[120.3, 120.3\] | 124.9 \[124.7, 124.9\] | 97.7 \[96.6, 98.3\] | 350.4 \[349.3, 351.0\] |
| epydemic | 165865 | 117.2 \[117.2, 117.2\] | 125.0 \[125.0, 125.0\] | 166.7 \[166.5, 166.9\] | 416.3 \[416.0, 416.4\] |
| Agents.jl | 165865 | 482.3 \[482.3, 482.5\] | 2.8 \[2.8, 2.9\] | 10.3 \[10.2, 10.4\] | 568.6 \[567.7, 569.2\] |
| FRED | 165865 |  |  |  | 784.4 \[784.4, 784.4\] |

![](README_files/figure-commonmark/memory-plot-1.png)

### Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show the final attack rate and peak
hospitalization load.

| Engine     | Agents | Median final attack rate | Median peak hospitalized |
|:-----------|-------:|-------------------------:|-------------------------:|
| Agents.jl  | 165865 |                    0.624 |                    855.0 |
| covasim    | 165865 |                    0.699 |                   1725.5 |
| EoN        | 165865 |                    0.620 |                   1004.0 |
| epiworld   | 165865 |                    0.622 |                    843.5 |
| epiworldpy | 165865 |                    0.622 |                    843.5 |
| epiworldR  | 165865 |                    0.622 |                    843.5 |
| epydemic   | 165865 |                    0.623 |                   1104.5 |
| FRED       | 165865 |                    0.631 |                   1110.5 |
| individual | 165865 |                    0.630 |                   1001.5 |
| ixa        | 165865 |                    0.625 |                    851.5 |
| starsim    | 165865 |                    0.612 |                    968.5 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

### Code required to define the model

A rough measure of effort: how much code each engine needs to express
the model. The counting rules are described in the [project
overview](../README.md#measuring-implementation-effort), and the counted
regions are listed in [`code_regions.yml`](code_regions.yml).

| Engine     | Language | Files | Model lines |
|:-----------|:---------|------:|------------:|
| epiworld   | C++      |     1 |          26 |
| individual | R        |     1 |          32 |
| epiworldpy | Python   |     1 |          35 |
| EoN        | Python   |     1 |          38 |
| epiworldR  | R        |     1 |          47 |
| covasim    | Python   |     1 |          65 |
| Agents.jl  | Julia    |     1 |          68 |
| epydemic   | Python   |     1 |          71 |
| starsim    | Python   |     1 |          84 |
| FRED       | Python   |     1 |          92 |
| ixa        | Rust     |     3 |         138 |

## Recorded versions

| Engine     | Recorded version   |
|:-----------|:-------------------|
| Agents.jl  | 7.0.4              |
| covasim    | 3.1.8              |
| EoN        | 1.92               |
| epiworld   | 0.17.1+g04c4ad8    |
| epiworldpy | 0.17.1-0+gc24b4f9  |
| epiworldR  | 0.17.1.0+gfca60f3  |
| epydemic   | 1.14.1             |
| FRED       | PUB.5.7.0+gbd25f04 |
| individual | 0.1.19             |
| ixa        | 3.1.0              |
| starsim    | 3.6.1              |
