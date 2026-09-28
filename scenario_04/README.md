# Scenario 04: SEIRH on a collapsed GeoPops network

2026-09-28

- [Model](#model)
  - [The network](#the-network)
  - [Calibration](#calibration)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
  - [The epiworld family](#the-epiworld-family)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
  - [Code required to define the
    model](#code-required-to-define-the-model)
- [Recorded versions](#recorded-versions)

[Back to the project overview](../README.md) · [Scenario 00
report](../scenario_00/README.md)

[Scenario 00](../scenario_00/README.md)’s model on a real contact
network instead of the shared Watts–Strogatz graph: the model, its
parameters, the 100 seed cases, and the runners in [`runners/`](runners)
are copies of scenario 00’s. Two engines are added here rather than in
scenario 00 because they need extra build steps and are only worth that
cost once: FRED (a compiled, population-and-network-driven simulator)
and Agents.jl (Julia). This scenario also asks a different question from
scenario 00: does the picture change on a graph with real degree
heterogeneity, rather than Watts–Strogatz’s near-constant degree?

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
undirected, unweighted graph: 376,526 edges, mean degree 3.77. Layer
identity, edge weights, demographic attributes, and contacts touching
excluded rows are discarded; only which pairs of the 200,000 people are
ever in contact survives. [`scenario.toml`](scenario.toml) records the
exact commit and per-layer checksums.

| Parameter (`scenario.toml`) | Value | Meaning |
|:---|---:|:---|
| Population | 200,000 | GeoPops rows kept |
| Edges | 376,526 | after union and dedup |
| Mean degree | 3.77 | observed, not fixed |
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

The nine engines from [scenario
00](../scenario_00/README.md#engine-implementations) run the same model
unchanged, reading this scenario’s edge list instead. Two more are
added:

- **FRED** (<a href="https://github.com/PublicHealthDynamicsLab/FRED"
  target="_blank">PublicHealthDynamicsLab/FRED</a>): a compiled,
  DSL-driven simulator with no edge-list network primitive comparable to
  the other engines’ graphs. The runner
  ([`run_FRED.py`](runners/run_FRED.py)) builds a synthetic population
  of 200,000 single-person households (so FRED’s own household mixing
  adds no contacts beyond the network) and loads the benchmark’s edges
  verbatim as `Contact.add_edge` properties on FRED’s `Network` group
  type, with transmission restricted to that network. Its contact model
  differs from the other engines’ constant per-edge hazard: each
  infectious agent draws a number of successful contacts per day
  proportional to its degree, so the runner derives a per-day, per-edge
  probability that reproduces the same target $R_0$ over the mean
  infectious period ([`daily_edge_probability`](runners/run_FRED.py)).
- **Agents.jl** (<a href="https://github.com/JuliaDynamics/Agents.jl"
  target="_blank">JuliaDynamics/Agents.jl</a>): a `StandardABM` with a
  per-agent status and a per-day `model_step!` that scans each
  infectious agent’s adjacency list, matching the synchronous daily
  semantics of epiworldR, epydemic, and individual. The adjacency list
  and transmission math are written for this benchmark, since Agents.jl
  has no built-in epidemic model.

**MEmilio is deliberately excluded.** Its ABM has no edge-list network
primitive either — agents interact only through shared Locations
(household, work, school, …) — so matching this network would need on
the order of 376,000 synthetic two-person Locations, of unproven
performance, on top of an 8-compartment, viral-load-driven disease model
with no simple SEIRH mapping. See [issue
\#11](https://github.com/UofUEpiBio/epiworld-benchmark/issues/11).

## Results

> [!WARNING]
>
> No full-profile results are present for this scenario yet. Run
> `make benchmark`, then render this report again.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 1                                                     |
| Latest run failures | 0                                                     |

    NULL

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

### Speed relative to epiworldR

Ratios are matched by replicate seed. Values above one mean that
epiworldR completed the simulation call faster.

    NULL

### Time to a first result

The simulation time above covers what an engine has to redo for every
replicate on the same network (see [what is
measured](../README.md#run-time)). *Build* is the rest of the setup:
turning the edge list into the engine’s population and network, which a
user does once per network. *Build + simulate* is the time to a first
result once the edge list is in memory.

    NULL

### The epiworld family

    NULL

### Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show the final attack rate and peak
hospitalization load.

    NULL

### Code required to define the model

A rough measure of effort: how much code each engine needs to express
the model. The counting rules are described in the [project
overview](../README.md#measuring-implementation-effort), and the counted
regions are listed in [`code_regions.yml`](code_regions.yml).

| Engine     | Language | Files | Model lines |
|:-----------|:---------|------:|------------:|
| epiworld   | C++      |     1 |          26 |
| individual | R        |     1 |          32 |
| epiworldpy | Python   |     1 |          34 |
| EoN        | Python   |     1 |          37 |
| epiworldR  | R        |     1 |          47 |
| Agents.jl  | Julia    |     1 |          54 |
| covasim    | Python   |     1 |          64 |
| epydemic   | Python   |     1 |          70 |
| starsim    | Python   |     1 |          83 |
| FRED       | Python   |     1 |          95 |
| ixa        | Rust     |     3 |         138 |

## Recorded versions

    NULL
