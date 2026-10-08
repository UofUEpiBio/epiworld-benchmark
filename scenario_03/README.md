# Scenario 03: SEIRH at 1,000,000 agents

2026-10-01

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
[Methods](../docs/methods.md) · [Results](../docs/results.md) ·
[Scenario 00](../scenario_00/README.md)

[Scenario 00](../scenario_00/README.md) at 1,000,000 agents, ten times
its largest size. It asks how each engine’s time grows with the
population when the outbreak does not: with 100 seed cases and the same
per-contact risk, about 6,100 agents are infected at 100,000 agents and
6,600 at 1,000,000 (median attack rates 0.061 and 0.0066). The
[results](../docs/results.md#scaling) compare the growth across engines.

## Model

The model, parameters, network settings, seed cases, and runners
([`runners/`](runners)) are scenario 00’s. Each engine runs 20
replicates rather than 100, because the slowest take tens of seconds per
replicate, and the `[design]` table in [`scenario.toml`](scenario.toml)
keeps this scenario sequential whatever `N_THREADS` says
([why](../docs/methods.md#execution-protocol)).

| Parameter                     | Value |
|:------------------------------|------:|
| `hospital_days`               |     7 |
| `hospitalization_probability` |  0.05 |
| `infectious_days`             |     7 |
| `initial_infected`            |   100 |
| `latent_days`                 |     4 |
| `target_r0`                   |     2 |

## Engine notes

None: every runner is scenario 00’s.

## Results

> [!TIP]
>
> The complete 240-run design for this scenario is available.

> [!WARNING]
>
> This scenario’s results were produced in 2 different environments (see
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
| Run dates | 2026-09-30 to 2026-10-01 |
| Latest run | 220 executed, 0 cached, 0 failed |

| Engine     |  Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|--------:|-----:|----------------------:|-------:|-------:|
| epiworldR  | 1000000 |   20 |                 0.024 |  0.023 |  0.024 |
| epiworld   | 1000000 |   20 |                 0.026 |  0.025 |  0.027 |
| epiworldpy | 1000000 |   20 |                 0.027 |  0.026 |  0.028 |
| individual | 1000000 |   20 |                 0.272 |  0.270 |  0.273 |
| ixa        | 1000000 |   20 |                 0.373 |  0.359 |  0.390 |
| Agents.jl  | 1000000 |   20 |                 0.727 |  0.688 |  0.802 |
| EoN\*      | 1000000 |   20 |                 2.266 |  2.243 |  2.298 |
| covasim    | 1000000 |   20 |                 3.144 |  3.088 |  3.241 |
| FRED       | 1000000 |   20 |                 7.204 |  6.811 |  7.757 |
| starsim    | 1000000 |   20 |                10.962 | 10.780 | 11.119 |
| ABM\*      | 1000000 |   20 |                10.963 | 10.200 | 11.635 |
| epydemic   | 1000000 |   20 |                13.333 | 13.246 | 13.499 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Simulation time

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by replicate seed. Values above one mean that
epiworldR completed the simulation call faster.

| Engine     |  Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|--------:|------------------------:|-------:|-------:|
| epiworld   | 1000000 |                    1.10 |   1.03 |   1.15 |
| epiworldpy | 1000000 |                    1.15 |   1.10 |   1.21 |
| covasim    | 1000000 |                  134.64 | 131.19 | 139.52 |
| starsim    | 1000000 |                  471.29 | 453.60 | 484.84 |
| EoN\*      | 1000000 |                   96.08 |  94.65 |  98.69 |
| epydemic   | 1000000 |                  571.62 | 552.84 | 587.43 |
| ixa        | 1000000 |                   16.08 |  15.31 |  16.51 |
| individual | 1000000 |                   11.62 |  11.25 |  11.84 |
| ABM\*      | 1000000 |                  466.83 | 418.59 | 506.21 |
| FRED       | 1000000 |                  309.84 | 291.92 | 325.40 |
| Agents.jl  | 1000000 |                   31.18 |  28.49 |  34.17 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Time to a first result

[Timing](../docs/methods.md#timing) defines build and simulate.

| Engine     |  Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|--------:|---------------:|----------:|-------------:|---------------------:|
| epiworld   | 1000000 |          0.492 |     0.127 |        0.026 |                0.153 |
| epiworldR  | 1000000 |          0.815 |     0.155 |        0.024 |                0.178 |
| ixa        | 1000000 |          0.288 |     0.003 |        0.373 |                0.376 |
| epiworldpy | 1000000 |          1.531 |     0.365 |        0.027 |                0.392 |
| individual | 1000000 |          0.823 |     0.263 |        0.272 |                0.534 |
| Agents.jl  | 1000000 |          1.808 |     0.170 |        0.727 |                0.905 |
| covasim    | 1000000 |          1.553 |     0.047 |        3.144 |                3.191 |
| EoN\*      | 1000000 |          1.543 |     3.607 |        2.266 |                5.866 |
| starsim    | 1000000 |          1.523 |     0.001 |       10.962 |               10.963 |
| ABM\*      | 1000000 |          0.950 |     0.349 |       10.963 |               11.331 |
| FRED       | 1000000 |         28.374 |     9.367 |        7.204 |               16.803 |
| epydemic   | 1000000 |          1.492 |     3.558 |       13.333 |               16.948 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Memory

Median \[Q1, Q3\] in MiB ([Memory](../docs/methods.md#memory)).

| Engine | Agents | Baseline (MiB) | Model footprint (MiB) | Simulation memory (MiB) | Overall peak (MiB) |
|:---|:---|---:|---:|---:|---:|
| individual | 1000000 | 74.8 \[74.8, 74.8\] | 7.0 \[7.0, 7.0\] | 125.7 \[125.3, 125.8\] | 282.3 \[282.3, 282.3\] |
| epiworld | 1000000 | 3.0 \[3.0, 3.0\] | 233.9 \[233.9, 233.9\] | 28.9 \[28.9, 28.9\] | 351.8 \[351.8, 351.8\] |
| ixa | 1000000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 329.9 \[329.9, 329.9\] | 408.7 \[408.7, 408.7\] |
| epiworldR | 1000000 | 72.1 \[72.1, 72.1\] | 210.5 \[210.5, 210.5\] | 0.0 \[0.0, 0.0\] | 478.6 \[478.5, 478.6\] |
| covasim | 1000000 | 225.1 \[225.1, 225.1\] | 16.2 \[16.1, 16.3\] | 288.8 \[288.6, 288.8\] | 684.5 \[684.5, 684.8\] |
| epiworldpy | 1000000 | 46.6 \[46.6, 46.6\] | 218.6 \[218.5, 218.6\] | 43.4 \[43.4, 43.4\] | 854.3 \[854.2, 854.3\] |
| starsim | 1000000 | 260.9 \[260.9, 260.9\] | 0.0 \[0.0, 0.0\] | 565.7 \[565.2, 566.0\] | 905.1 \[905.1, 905.2\] |
| Agents.jl | 1000000 | 481.9 \[481.7, 482.1\] | 34.0 \[30.2, 41.9\] | 8.4 \[8.4, 8.7\] | 970.7 \[955.8, 975.0\] |
| ABM\* | 1000000 | 74.4 \[74.4, 74.4\] | 7.0 \[7.0, 7.0\] | 1128.6 \[1126.5, 1129.5\] | 1343.6 \[1341.8, 1344.3\] |
| EoN\* | 1000000 | 120.3 \[120.3, 120.3\] | 1182.4 \[1182.4, 1182.5\] | 290.0 \[290.0, 290.0\] | 1670.8 \[1670.8, 1670.8\] |
| epydemic | 1000000 | 117.2 \[117.2, 117.2\] | 1181.0 \[1181.0, 1181.7\] | 1642.8 \[1642.8, 1642.8\] | 3020.0 \[3020.0, 3020.0\] |
| FRED | 1000000 |  |  |  | 5846.8 \[5846.7, 5846.8\] |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Epidemiological sanity checks

| Engine     |  Agents | Median final attack rate | Median peak hospitalized |
|:-----------|--------:|-------------------------:|-------------------------:|
| ABM\*      | 1000000 |                    0.006 |                     33.5 |
| Agents.jl  | 1000000 |                    0.007 |                     34.5 |
| covasim    | 1000000 |                    0.007 |                     38.0 |
| EoN\*      | 1000000 |                    0.007 |                     41.0 |
| epiworld   | 1000000 |                    0.006 |                     32.0 |
| epiworldpy | 1000000 |                    0.006 |                     32.0 |
| epiworldR  | 1000000 |                    0.006 |                     32.0 |
| epydemic   | 1000000 |                    0.007 |                     43.5 |
| FRED       | 1000000 |                    0.006 |                     34.0 |
| individual | 1000000 |                    0.006 |                     34.0 |
| ixa        | 1000000 |                    0.006 |                     30.5 |
| starsim    | 1000000 |                    0.007 |                     40.0 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Model lines

The runners are scenario 00’s, so are their model lines (see [scenario
00](../scenario_00/README.md#model-lines)).

## Notes

**Calibration.** The factors were not recalibrated at this size. Each
engine reuses its 100,000-agent factor from scenario 00, which barely
moved between 10,000 and 100,000 agents, and the attack rates above stay
aligned.

**Memory sets the limits here.** At this size some engines need several
gigabytes per replicate (see the memory table), which is why four
concurrent replicates competed for memory and why this scenario runs one
at a time.
