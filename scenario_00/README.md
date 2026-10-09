# Scenario 00: SEIRH baseline

2026-10-09

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

The reference model for the benchmark. Every later scenario adds
complexity to it, and the [results](../docs/results.md) compare each
later scenario with this one.

## Model

The [common model](../docs/methods.md#common-model) with no
interventions, on the shared Watts–Strogatz network (mean degree 10,
rewiring probability 0.05) at 10,000 and 100,000 agents, with 100
replicates per engine and size.

| Parameter                     | Value |
|:------------------------------|------:|
| `hospital_days`               |     7 |
| `hospitalization_probability` |  0.05 |
| `infectious_days`             |     7 |
| `initial_infected`            |   100 |
| `latent_days`                 |     4 |
| `target_r0`                   |     2 |

## Engine notes

Every engine runs its baseline implementation, described in
[Engines](../docs/methods.md#engines). The three epiworld runners build
the same model and, in this scenario, produce identical epidemics for
every seed.

## Results

> [!TIP]
>
> The complete 2,600-run design for this scenario is available.

> [!WARNING]
>
> This scenario’s results were produced in 3 different environments (see
> `environment_id` in `results/results.csv`). The table describes the
> latest run only, and timings from different environments are not
> comparable.

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
| Latest run | 2200 executed, 0 cached, 0 failed |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| epiworldR  |  10000 |  100 |                 0.004 |  0.004 |  0.005 |
| epiworldpy |  10000 |  100 |                 0.004 |  0.004 |  0.004 |
| epiworld   |  10000 |  100 |                 0.004 |  0.004 |  0.005 |
| ixa        |  10000 |  100 |                 0.007 |  0.006 |  0.007 |
| Agents.jl  |  10000 |  100 |                 0.020 |  0.019 |  0.021 |
| individual |  10000 |  100 |                 0.024 |  0.024 |  0.025 |
| covasim    |  10000 |  100 |                 0.062 |  0.062 |  0.063 |
| FRED       |  10000 |  100 |                 0.066 |  0.063 |  0.070 |
| EoN\*      |  10000 |  100 |                 0.076 |  0.072 |  0.080 |
| ABM\*      |  10000 |  100 |                 0.140 |  0.134 |  0.151 |
| starsim    |  10000 |  100 |                 0.165 |  0.163 |  0.166 |
| epydemic   |  10000 |  100 |                 0.489 |  0.469 |  0.515 |
| EpiModel   |  10000 |  100 |                 0.651 |  0.631 |  0.669 |
| epiworldpy | 100000 |  100 |                 0.007 |  0.007 |  0.008 |
| epiworld   | 100000 |  100 |                 0.008 |  0.007 |  0.008 |
| epiworldR  | 100000 |  100 |                 0.008 |  0.008 |  0.009 |
| ixa        | 100000 |  100 |                 0.033 |  0.033 |  0.034 |
| Agents.jl  | 100000 |  100 |                 0.037 |  0.036 |  0.037 |
| individual | 100000 |  100 |                 0.053 |  0.052 |  0.054 |
| EoN\*      | 100000 |  100 |                 0.285 |  0.275 |  0.294 |
| covasim    | 100000 |  100 |                 0.324 |  0.323 |  0.334 |
| FRED       | 100000 |  100 |                 0.485 |  0.475 |  0.499 |
| starsim    | 100000 |  100 |                 0.905 |  0.897 |  0.924 |
| ABM\*      | 100000 |  100 |                 0.936 |  0.883 |  1.006 |
| epydemic   | 100000 |  100 |                 1.706 |  1.628 |  1.790 |
| EpiModel   | 100000 |  100 |                 7.025 |  6.934 |  7.274 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Simulation time

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|-------:|------------------------:|-------:|-------:|
| epiworld   |  10000 |                    0.95 |   0.91 |   1.02 |
| epiworldpy |  10000 |                    0.91 |   0.85 |   0.97 |
| covasim    |  10000 |                   15.24 |  12.49 |  15.57 |
| starsim    |  10000 |                   40.20 |  33.00 |  41.13 |
| EoN\*      |  10000 |                   17.24 |  15.17 |  19.10 |
| epydemic   |  10000 |                  113.92 |  98.02 | 122.41 |
| ixa        |  10000 |                    1.47 |   1.33 |   1.70 |
| individual |  10000 |                    5.75 |   4.80 |   6.00 |
| ABM\*      |  10000 |                   33.00 |  28.75 |  35.00 |
| EpiModel   |  10000 |                  152.25 | 130.60 | 162.75 |
| FRED       |  10000 |                   15.08 |  13.15 |  16.83 |
| Agents.jl  |  10000 |                    4.80 |   3.95 |   4.96 |
| epiworld   | 100000 |                    0.95 |   0.91 |   0.98 |
| epiworldpy | 100000 |                    0.90 |   0.86 |   0.95 |
| covasim    | 100000 |                   40.44 |  36.39 |  44.29 |
| starsim    | 100000 |                  113.11 | 100.95 | 123.39 |
| EoN\*      | 100000 |                   34.89 |  31.84 |  38.37 |
| epydemic   | 100000 |                  209.15 | 194.04 | 230.55 |
| ixa        | 100000 |                    4.14 |   3.75 |   4.64 |
| individual | 100000 |                    6.63 |   5.89 |   7.43 |
| ABM\*      | 100000 |                  118.04 | 106.91 | 131.18 |
| EpiModel   | 100000 |                  875.19 | 804.36 | 971.21 |
| FRED       | 100000 |                   60.48 |  54.45 |  67.25 |
| Agents.jl  | 100000 |                    4.57 |   4.12 |   5.01 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Time to a first result

*Build* turns the edge list into the engine’s model, once per network;
*build + simulate* is the time to a first result once the edge list is
in memory ([Timing](../docs/methods.md#timing)).

| Engine     | Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|-------:|---------------:|----------:|-------------:|---------------------:|
| epiworld   |  10000 |          0.005 |     0.001 |        0.004 |                0.005 |
| ixa        |  10000 |          0.003 |     0.000 |        0.007 |                0.007 |
| epiworldpy |  10000 |          0.016 |     0.003 |        0.004 |                0.008 |
| epiworldR  |  10000 |          0.007 |     0.016 |        0.004 |                0.020 |
| Agents.jl  |  10000 |          0.138 |     0.020 |        0.020 |                0.040 |
| individual |  10000 |          0.007 |     0.018 |        0.024 |                0.042 |
| covasim    |  10000 |          0.016 |     0.037 |        0.062 |                0.099 |
| EoN\*      |  10000 |          0.016 |     0.028 |        0.076 |                0.104 |
| ABM\*      |  10000 |          0.007 |     0.023 |        0.140 |                0.164 |
| starsim    |  10000 |          0.015 |     0.000 |        0.165 |                0.165 |
| FRED       |  10000 |          0.242 |     0.109 |        0.066 |                0.175 |
| epydemic   |  10000 |          0.015 |     0.017 |        0.489 |                0.507 |
| EpiModel   |  10000 |          0.008 |     0.207 |        0.651 |                0.857 |
| epiworld   | 100000 |          0.048 |     0.011 |        0.008 |                0.018 |
| ixa        | 100000 |          0.028 |     0.000 |        0.033 |                0.034 |
| epiworldR  | 100000 |          0.068 |     0.028 |        0.008 |                0.036 |
| epiworldpy | 100000 |          0.152 |     0.033 |        0.007 |                0.041 |
| Agents.jl  | 100000 |          0.273 |     0.025 |        0.037 |                0.062 |
| individual | 100000 |          0.068 |     0.030 |        0.053 |                0.083 |
| covasim    | 100000 |          0.155 |     0.039 |        0.324 |                0.363 |
| EoN\*      | 100000 |          0.152 |     0.241 |        0.285 |                0.526 |
| starsim    | 100000 |          0.150 |     0.000 |        0.905 |                0.905 |
| ABM\*      | 100000 |          0.075 |     0.041 |        0.936 |                0.982 |
| FRED       | 100000 |          2.413 |     0.798 |        0.485 |                1.283 |
| epydemic   | 100000 |          0.149 |     0.277 |        1.706 |                1.990 |
| EpiModel   | 100000 |          0.070 |     0.653 |        7.025 |                7.693 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Memory

Median \[Q1, Q3\] in MiB ([Memory](../docs/methods.md#memory)).

| Engine | Agents | Baseline (MiB) | Model footprint (MiB) | Simulation memory (MiB) | Overall peak (MiB) |
|:---|:---|---:|---:|---:|---:|
| ixa | 10000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 3.1 \[3.1, 3.3\] | 6.5 \[6.5, 6.6\] |
| epiworld | 10000 | 3.0 \[3.0, 3.0\] | 2.3 \[2.3, 2.3\] | 2.0 \[1.9, 2.1\] | 8.4 \[8.2, 8.5\] |
| epiworldpy | 10000 | 46.6 \[46.6, 46.6\] | 2.6 \[2.6, 2.6\] | 0.8 \[0.7, 0.9\] | 54.8 \[54.8, 54.8\] |
| FRED | 10000 |  |  |  | 68.3 \[68.2, 68.3\] |
| epiworldR | 10000 | 72.1 \[72.1, 72.1\] | 2.7 \[2.7, 2.7\] | 0.0 \[0.0, 0.0\] | 75.6 \[75.5, 75.6\] |
| ABM\* | 10000 | 74.4 \[74.4, 74.4\] | 1.9 \[1.9, 1.9\] | 16.2 \[16.1, 16.4\] | 93.1 \[93.0, 93.2\] |
| individual | 10000 | 74.8 \[74.8, 74.8\] | 1.5 \[1.5, 1.5\] | 13.9 \[13.4, 14.4\] | 93.4 \[93.1, 93.9\] |
| EoN\* | 10000 | 120.3 \[120.3, 120.3\] | 10.1 \[10.1, 10.3\] | 3.1 \[3.0, 3.4\] | 136.6 \[136.4, 136.8\] |
| epydemic | 10000 | 117.2 \[117.2, 117.2\] | 10.1 \[10.1, 10.1\] | 15.6 \[15.6, 15.8\] | 145.9 \[145.9, 146.0\] |
| covasim | 10000 | 225.1 \[225.1, 225.1\] | 1.4 \[1.4, 1.4\] | 4.8 \[4.6, 4.9\] | 234.5 \[234.5, 234.6\] |
| starsim | 10000 | 260.9 \[260.9, 260.9\] | 0.0 \[0.0, 0.0\] | 6.7 \[6.6, 6.8\] | 270.5 \[270.5, 270.7\] |
| EpiModel | 10000 | 231.3 \[231.3, 231.3\] | 16.2 \[16.2, 16.2\] | 23.2 \[22.9, 35.8\] | 272.0 \[271.2, 284.5\] |
| Agents.jl | 10000 | 482.6 \[482.3, 483.0\] | 2.0 \[2.0, 2.0\] | 0.5 \[0.5, 0.5\] | 518.1 \[517.6, 519.8\] |
| epiworld | 100000 | 3.0 \[3.0, 3.0\] | 23.3 \[23.3, 23.3\] | 4.5 \[4.2, 4.8\] | 38.1 \[38.1, 38.1\] |
| ixa | 100000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 32.0 \[32.0, 32.0\] | 42.2 \[42.2, 42.3\] |
| epiworldR | 100000 | 72.1 \[72.1, 72.1\] | 23.4 \[23.4, 23.4\] | 0.0 \[0.0, 0.0\] | 113.2 \[113.2, 113.2\] |
| individual | 100000 | 74.8 \[74.8, 74.8\] | 6.2 \[6.2, 6.2\] | 35.1 \[33.8, 35.5\] | 127.5 \[126.3, 127.9\] |
| epiworldpy | 100000 | 46.7 \[46.6, 46.7\] | 34.6 \[34.6, 34.6\] | 0.0 \[0.0, 0.0\] | 127.5 \[127.5, 127.5\] |
| ABM\* | 100000 | 74.4 \[74.4, 74.4\] | 6.2 \[6.2, 6.2\] | 98.1 \[98.0, 98.5\] | 192.4 \[192.3, 192.8\] |
| covasim | 100000 | 225.1 \[225.1, 225.1\] | 2.0 \[2.0, 2.2\] | 33.1 \[32.9, 33.2\] | 272.0 \[272.0, 272.0\] |
| EoN\* | 100000 | 120.3 \[120.3, 120.3\] | 126.1 \[125.9, 126.5\] | 27.8 \[27.8, 27.8\] | 283.6 \[283.6, 283.6\] |
| starsim | 100000 | 260.9 \[260.9, 260.9\] | 0.0 \[0.0, 0.0\] | 67.6 \[66.5, 68.3\] | 339.0 \[337.5, 339.6\] |
| epydemic | 100000 | 117.2 \[117.2, 117.2\] | 126.2 \[126.2, 126.4\] | 160.5 \[160.5, 160.5\] | 413.2 \[413.2, 413.3\] |
| EpiModel | 100000 | 231.3 \[231.3, 231.3\] | 137.1 \[137.1, 137.1\] | 105.7 \[105.7, 105.7\] | 485.0 \[485.0, 485.0\] |
| Agents.jl | 100000 | 482.5 \[482.4, 482.8\] | 10.8 \[10.6, 10.9\] | 1.5 \[1.4, 1.5\] | 574.3 \[574.0, 574.5\] |
| FRED | 100000 |  |  |  | 581.0 \[580.9, 581.0\] |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Epidemiological sanity checks

| Engine     | Agents | Median final attack rate | Median peak hospitalized |
|:-----------|-------:|-------------------------:|-------------------------:|
| ABM\*      |  10000 |                    0.382 |                     22.0 |
| Agents.jl  |  10000 |                    0.387 |                     18.0 |
| covasim    |  10000 |                    0.392 |                     23.0 |
| EoN\*      |  10000 |                    0.379 |                     21.0 |
| EpiModel   |  10000 |                    0.387 |                     22.0 |
| epiworld   |  10000 |                    0.385 |                     19.0 |
| epiworldpy |  10000 |                    0.385 |                     19.0 |
| epiworldR  |  10000 |                    0.385 |                     19.0 |
| epydemic   |  10000 |                    0.396 |                     25.0 |
| FRED       |  10000 |                    0.385 |                     21.0 |
| individual |  10000 |                    0.381 |                     20.5 |
| ixa        |  10000 |                    0.378 |                     18.5 |
| starsim    |  10000 |                    0.387 |                     21.0 |
| ABM\*      | 100000 |                    0.059 |                     33.0 |
| Agents.jl  | 100000 |                    0.062 |                     30.0 |
| covasim    | 100000 |                    0.061 |                     35.0 |
| EoN\*      | 100000 |                    0.060 |                     32.0 |
| EpiModel   | 100000 |                    0.060 |                     34.0 |
| epiworld   | 100000 |                    0.060 |                     30.0 |
| epiworldpy | 100000 |                    0.060 |                     30.0 |
| epiworldR  | 100000 |                    0.060 |                     30.0 |
| epydemic   | 100000 |                    0.064 |                     41.0 |
| FRED       | 100000 |                    0.060 |                     33.0 |
| individual | 100000 |                    0.060 |                     34.0 |
| ixa        | 100000 |                    0.062 |                     31.0 |
| starsim    | 100000 |                    0.059 |                     33.5 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

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

**Calibration.** epiworldR, its two siblings, EoN, epydemic, Starsim,
and ABM run uncalibrated: their medians already agree. The others use
size-specific factors at 10,000 and 100,000 agents (in
[`scenario.toml`](scenario.toml)): Covasim 0.715 and 0.710, ixa 0.983
and 0.988, individual 0.951 at both, FRED 0.911 and 0.898, Agents.jl
0.981 and 0.982, and EpiModel 0.948 and 0.940. Their uncalibrated
medians sat above the target: ixa’s, individual’s (0.441 and 0.074), and
Agents.jl’s (0.414 and 0.065) because they seed the initial cases
infectious, individual’s also because its infectious period averages the
full seven days, where the epiworld family’s competing hospitalization
rate shortens it; FRED’s (0.519 and 0.095) because each of its
generations is about a day shorter, and EpiModel’s (0.458 and 0.086)
because its progression module runs before its infection module, so a
new infectious case transmits on the day it turns infectious. Covasim’s
were calibrated after restricting it to fixed transmission and permanent
immunity.

**Model lines.** They vary with how much of the model an engine
provides. EoN describes transitions as rate graphs, Covasim overrides
its native parameters to restrict itself to SEIRH, Starsim subclasses
its SEIR for the hospital state, the ixa, individual, and Agents.jl
runners write their daily step or infection process by hand, and the
EpiModel runner writes its infection, progression, and prevalence
modules. The FRED runner’s lines are the Python that writes FRED’s model
file and population.
