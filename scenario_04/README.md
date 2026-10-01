# Scenario 04: SEIRH on a GeoPops contact network

2026-10-01

- [Model](#model)
  - [The network](#the-network)
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
[Methods](../docs/methods.md) · [Results](../docs/results.md) ·
[Scenario 00](../scenario_00/README.md)

[Scenario 00](../scenario_00/README.md)’s model on a real contact
network instead of the shared Watts–Strogatz graph. The model, its
parameters, the 100 seed cases, and the runners in [`runners/`](runners)
are scenario 00’s. It asks whether the picture changes on a graph with
real degree heterogeneity rather than Watts–Strogatz’s near-constant
degree.

## Model

### The network

The network comes from
<a href="https://github.com/GeoPopsHub" target="_blank">GeoPops</a>’
<a href="https://github.com/GeoPopsHub/sc_spartanburg_measles"
target="_blank">Spartanburg County, South Carolina synthetic
population</a>: a real, degree-heterogeneous contact structure, at the
cost of not being able to vary population size independently of the
source data.
[`scripts/generate_network.py`](../scripts/generate_network.py)
downloads GeoPops’ four contact-layer adjacency matrices (household,
workplace, school, and group quarters, each a checksum-verified
`MatrixMarket` file pinned to a commit), keeps the induced subgraph on
the first 200,000 rows of each, and unions the four layers into one
undirected, unweighted graph. Of those 200,000 people, 34,135 (17%) have
no contacts among the others; they could never be infected, so they are
dropped and the remaining 165,865 renumbered in row order. Layer
identity, edge weights, demographic attributes, and contacts touching
excluded rows are discarded; only which pairs of people are ever in
contact survives. [`scenario.toml`](scenario.toml) records the exact
commit and per-layer checksums. This is a collapsed contact graph, not a
reproduction of GeoPops’ multilayer, demographic model.

| Property                             |                        Value |
|:-------------------------------------|-----------------------------:|
| Agents                               |                      165,865 |
| Edges, after union and deduplication |                      376,526 |
| Mean degree (observed)               |                         4.54 |
| Degree range (median)                |                  1 to 25 (3) |
| Components (largest)                 | 17,128 (127,291 agents, 77%) |
| Replicates per engine                |                           20 |

The largest component caps the attack rate. Because households form
cliques and degrees vary, the shared transmission mapping, which divides
$R_0$ by the mean degree minus one, does not give an $R_0$ of 2 here:
the mean excess degree is 6.9, not 3.5. The mapping is the same for
every engine, so the comparison stays aligned, but the outbreaks are
larger than the nominal $R_0$ suggests. Like scenario 03, this scenario
runs one replicate at a time (`workers = 1` in its `[design]` table).

| Parameter                     | Value |
|:------------------------------|------:|
| `hospital_days`               |     7 |
| `hospitalization_probability` |  0.05 |
| `infectious_days`             |     7 |
| `initial_infected`            |   100 |
| `latent_days`                 |     4 |
| `target_r0`                   |     2 |

## Engine notes

None: every engine runs scenario 00’s runner on this edge list.

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

[Timing](../docs/methods.md#timing) defines build and simulate.

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

### Memory

Median \[Q1, Q3\] in MiB ([Memory](../docs/methods.md#memory)).

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

### Epidemiological sanity checks

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

### Model lines

The runners are scenario 00’s, so are their model lines (see [scenario
00](../scenario_00/README.md#model-lines)).

## Notes

**Calibration.** Every engine runs uncalibrated here (a transmission
multiplier of 1): scenario 00’s factors were derived on the
Watts–Strogatz network at other sizes and do not carry over to this
topology. Most engines’ medians agree anyway; Covasim’s sits highest
(see the [results](../docs/results.md#epidemiological-agreement)). A
full run’s medians now exist to calibrate against, if aligned factors
are wanted.
