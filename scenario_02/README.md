# Scenario 02: vaccine plus epidemiological outputs

2026-10-09

- [Model](#model)
- [Engine notes](#engine-notes)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
  - [Extracting the outputs](#extracting-the-outputs)
  - [Memory](#memory)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
  - [Model lines](#model-lines)
- [Notes](#notes)

[Back to the project overview](../README.md) ·
[Methods](../docs/methods.md) · [Results](../docs/results.md) ·
[Scenario 01](../scenario_01/README.md)

[Scenario 01](../scenario_01/README.md), unchanged, plus four outputs
that every engine has to produce from each run, so the comparison
includes for every engine the bookkeeping epiworld always does. The
[results](../docs/results.md#cost-of-added-features) compare each
replicate with scenario 01’s, which runs the same epidemic for every
engine but individual: its runner picks each case’s source with R’s
global random numbers, so only its distribution of outcomes matches.

## Model

| Output | Definition |
|:---|:---|
| Transmission tree | The day, source, and target of every infection. Seed cases have no source. |
| Daily incidence | New infections (S → E) on each day from 1 to 100. |
| Reproductive number | epiworld’s definition: each case’s number of secondary infections, averaged over the cases infected on each day (a *case* reproductive number, so it falls toward zero near day 100). Day 0 holds the seed cases. |
| Transition matrix | The number of agents moving between each pair of S, E, I, H, and R on each day; day 0 is left out. |

Runners hold the outputs in memory, with the engine’s own tools where it
has them. Recording during the run is part of `simulate_seconds`;
reading the outputs out afterwards is `extract_seconds`. Records add
`extract_seconds`, `transmissions`, the transition matrix summed over
days 1 to 100 (`transitions_se`, …), and the daily series (medians in
`results/daily.csv`).

## Engine notes

- **The epiworld family**: nothing to add; the database already holds
  all four outputs (`get_transmissions()`,
  `get_hist_transition_matrix()`, and the reproductive-number getters).
  The C++ and Python runners aggregate the raw vectors themselves.
- **Covasim** reads its infection log; **Starsim** adds
  `ss.infection_log`. Both runners count each agent’s dated transitions
  within the run.
- **EoN**: `return_full_data=True` records each node’s status history,
  from which the runner builds the matrix, binning events into days.
- **epydemic** and **Agents.jl**: the model’s event handlers or daily
  step append each transmission and transition to lists.
- **ixa**: a data plugin, written by the daily step and by a
  `PropertyChangeEvent` subscription.
- **individual**: the infection process picks each source among the
  infectious neighbours; a second process compares the states on
  consecutive days.
- **EpiModel**: its transmission matrix (`save.transmat`, filled with
  `set_transmat()` by the infection module) and its `infTime` attribute
  for the tree, and an epidemic statistic per transition, set by the
  modules.
- **FRED**: its health records log every exposure and state change
  during the run; the runner parses them afterwards.

The runners other than the epiworld family compute the reproductive
number from the tree with epiworld’s definition, in a small helper
counted as model code.

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
| epiworldR  |  10000 |  100 |                 0.002 |  0.002 |  0.003 |
| epiworldpy |  10000 |  100 |                 0.002 |  0.002 |  0.002 |
| epiworld   |  10000 |  100 |                 0.002 |  0.002 |  0.002 |
| ixa        |  10000 |  100 |                 0.004 |  0.004 |  0.005 |
| Agents.jl  |  10000 |  100 |                 0.019 |  0.019 |  0.020 |
| individual |  10000 |  100 |                 0.040 |  0.040 |  0.041 |
| EoN\*      |  10000 |  100 |                 0.056 |  0.054 |  0.058 |
| covasim    |  10000 |  100 |                 0.061 |  0.061 |  0.062 |
| FRED       |  10000 |  100 |                 0.069 |  0.067 |  0.070 |
| ABM\*      |  10000 |  100 |                 0.099 |  0.095 |  0.106 |
| starsim    |  10000 |  100 |                 0.171 |  0.169 |  0.174 |
| epydemic   |  10000 |  100 |                 0.203 |  0.193 |  0.211 |
| EpiModel   |  10000 |  100 |                 0.615 |  0.604 |  0.633 |
| epiworldpy | 100000 |  100 |                 0.006 |  0.006 |  0.006 |
| epiworldR  | 100000 |  100 |                 0.007 |  0.006 |  0.007 |
| epiworld   | 100000 |  100 |                 0.007 |  0.007 |  0.007 |
| ixa        | 100000 |  100 |                 0.028 |  0.028 |  0.029 |
| Agents.jl  | 100000 |  100 |                 0.037 |  0.036 |  0.037 |
| individual | 100000 |  100 |                 0.070 |  0.069 |  0.071 |
| covasim    | 100000 |  100 |                 0.334 |  0.333 |  0.344 |
| EoN\*      | 100000 |  100 |                 0.450 |  0.445 |  0.468 |
| FRED       | 100000 |  100 |                 0.706 |  0.695 |  0.720 |
| starsim    | 100000 |  100 |                 0.715 |  0.709 |  0.743 |
| ABM\*      | 100000 |  100 |                 0.827 |  0.802 |  0.868 |
| epydemic   | 100000 |  100 |                 1.076 |  1.064 |  1.093 |
| EpiModel   | 100000 |  100 |                 7.665 |  7.514 |  7.850 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Simulation time

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |      Q1 |      Q3 |
|:-----------|-------:|------------------------:|--------:|--------:|
| epiworld   |  10000 |                    1.03 |    0.83 |    1.11 |
| epiworldpy |  10000 |                    0.92 |    0.77 |    1.04 |
| covasim    |  10000 |                   30.46 |   21.77 |   30.99 |
| starsim    |  10000 |                   84.61 |   59.49 |   86.67 |
| EoN\*      |  10000 |                   27.38 |   19.77 |   28.87 |
| epydemic   |  10000 |                   98.02 |   73.52 |  104.79 |
| ixa        |  10000 |                    2.20 |    1.61 |    2.32 |
| individual |  10000 |                   20.00 |   13.92 |   20.50 |
| ABM\*      |  10000 |                   48.50 |   37.50 |   51.62 |
| EpiModel   |  10000 |                  302.50 |  223.83 |  314.12 |
| FRED       |  10000 |                   33.57 |   24.07 |   34.76 |
| Agents.jl  |  10000 |                    9.57 |    7.05 |    9.73 |
| epiworld   | 100000 |                    1.08 |    1.02 |    1.18 |
| epiworldpy | 100000 |                    0.91 |    0.83 |    1.02 |
| covasim    | 100000 |                   49.45 |   47.52 |   55.64 |
| starsim    | 100000 |                  115.69 |  101.75 |  118.69 |
| EoN\*      | 100000 |                   69.53 |   63.89 |   75.54 |
| epydemic   | 100000 |                  157.56 |  152.49 |  179.41 |
| ixa        | 100000 |                    4.17 |    4.02 |    4.74 |
| individual | 100000 |                   10.21 |   10.00 |   11.71 |
| ABM\*      | 100000 |                  126.43 |  115.82 |  137.83 |
| EpiModel   | 100000 |                 1162.00 | 1082.64 | 1276.42 |
| FRED       | 100000 |                  103.89 |   99.67 |  117.77 |
| Agents.jl  | 100000 |                    5.70 |    5.21 |    6.10 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Time to a first result

[Timing](../docs/methods.md#timing) defines build and simulate.

| Engine     | Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|-------:|---------------:|----------:|-------------:|---------------------:|
| epiworld   |  10000 |          0.005 |     0.001 |        0.002 |                0.003 |
| ixa        |  10000 |          0.003 |     0.000 |        0.004 |                0.005 |
| epiworldpy |  10000 |          0.016 |     0.007 |        0.002 |                0.009 |
| epiworldR  |  10000 |          0.007 |     0.016 |        0.002 |                0.018 |
| Agents.jl  |  10000 |          0.174 |     0.024 |        0.019 |                0.043 |
| individual |  10000 |          0.007 |     0.017 |        0.040 |                0.058 |
| EoN\*      |  10000 |          0.016 |     0.028 |        0.056 |                0.084 |
| covasim    |  10000 |          0.016 |     0.038 |        0.061 |                0.099 |
| ABM\*      |  10000 |          0.007 |     0.021 |        0.099 |                0.120 |
| starsim    |  10000 |          0.016 |     0.000 |        0.171 |                0.171 |
| FRED       |  10000 |          0.240 |     0.128 |        0.069 |                0.197 |
| epydemic   |  10000 |          0.015 |     0.016 |        0.203 |                0.219 |
| EpiModel   |  10000 |          0.008 |     0.197 |        0.615 |                0.814 |
| epiworld   | 100000 |          0.049 |     0.011 |        0.007 |                0.018 |
| ixa        | 100000 |          0.028 |     0.000 |        0.028 |                0.029 |
| epiworldR  | 100000 |          0.070 |     0.030 |        0.007 |                0.037 |
| epiworldpy | 100000 |          0.151 |     0.041 |        0.006 |                0.047 |
| Agents.jl  | 100000 |          0.285 |     0.029 |        0.037 |                0.065 |
| individual | 100000 |          0.068 |     0.030 |        0.070 |                0.100 |
| covasim    | 100000 |          0.152 |     0.040 |        0.334 |                0.373 |
| EoN\*      | 100000 |          0.153 |     0.243 |        0.450 |                0.695 |
| starsim    | 100000 |          0.153 |     0.000 |        0.715 |                0.715 |
| ABM\*      | 100000 |          0.070 |     0.036 |        0.827 |                0.863 |
| epydemic   | 100000 |          0.147 |     0.276 |        1.076 |                1.357 |
| FRED       | 100000 |          2.419 |     0.825 |        0.706 |                1.535 |
| EpiModel   | 100000 |          0.070 |     0.646 |        7.665 |                8.322 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Extracting the outputs

Time to read the four outputs out of each engine after the run, not part
of the simulation time.

| Engine | Agents | Median simulation (s) | Median extraction (s) | Median extraction / simulation |
|:---|---:|---:|---:|---:|
| epiworldR | 10000 | 0.002 | 0.025 | 12.00 |
| epiworldpy | 10000 | 0.002 | 0.000 | 0.18 |
| epiworld | 10000 | 0.002 | 0.000 | 0.09 |
| ixa | 10000 | 0.004 | 0.000 | 0.02 |
| Agents.jl | 10000 | 0.019 | 0.006 | 0.32 |
| individual | 10000 | 0.040 | 0.001 | 0.03 |
| EoN\* | 10000 | 0.056 | 0.001 | 0.02 |
| covasim | 10000 | 0.061 | 0.000 | 0.00 |
| FRED | 10000 | 0.069 | 0.012 | 0.18 |
| ABM\* | 10000 | 0.099 | 0.001 | 0.01 |
| starsim | 10000 | 0.171 | 0.001 | 0.01 |
| epydemic | 10000 | 0.203 | 0.000 | 0.00 |
| EpiModel | 10000 | 0.615 | 0.045 | 0.07 |
| epiworldpy | 100000 | 0.006 | 0.000 | 0.08 |
| epiworldR | 100000 | 0.007 | 0.026 | 4.00 |
| epiworld | 100000 | 0.007 | 0.000 | 0.03 |
| ixa | 100000 | 0.028 | 0.000 | 0.00 |
| Agents.jl | 100000 | 0.037 | 0.007 | 0.19 |
| individual | 100000 | 0.070 | 0.001 | 0.01 |
| covasim | 100000 | 0.334 | 0.001 | 0.00 |
| EoN\* | 100000 | 0.450 | 0.007 | 0.02 |
| FRED | 100000 | 0.706 | 0.095 | 0.13 |
| starsim | 100000 | 0.715 | 0.001 | 0.00 |
| ABM\* | 100000 | 0.827 | 0.002 | 0.00 |
| epydemic | 100000 | 1.076 | 0.000 | 0.00 |
| EpiModel | 100000 | 7.665 | 0.050 | 0.01 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Memory

Median \[Q1, Q3\] in MiB ([Memory](../docs/methods.md#memory)).

| Engine | Agents | Baseline (MiB) | Model footprint (MiB) | Simulation memory (MiB) | Overall peak (MiB) |
|:---|:---|---:|---:|---:|---:|
| ixa | 10000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 3.1 \[3.1, 3.1\] | 6.4 \[6.4, 6.4\] |
| epiworld | 10000 | 2.9 \[2.9, 2.9\] | 2.5 \[2.5, 2.5\] | 1.8 \[1.8, 1.8\] | 8.5 \[8.5, 8.6\] |
| epiworldpy | 10000 | 46.7 \[46.7, 46.7\] | 7.6 \[7.6, 7.6\] | 0.5 \[0.5, 0.6\] | 59.4 \[59.4, 59.4\] |
| FRED | 10000 |  |  |  | 70.0 \[70.0, 70.0\] |
| epiworldR | 10000 | 72.1 \[72.1, 72.1\] | 2.8 \[2.8, 2.8\] | 0.0 \[0.0, 0.0\] | 76.1 \[76.1, 76.2\] |
| ABM\* | 10000 | 74.4 \[74.4, 74.4\] | 1.9 \[1.9, 1.9\] | 15.6 \[15.6, 15.7\] | 92.9 \[92.9, 93.1\] |
| individual | 10000 | 74.8 \[74.8, 74.8\] | 1.5 \[1.5, 1.5\] | 16.0 \[15.9, 16.2\] | 93.3 \[93.2, 93.6\] |
| EoN\* | 10000 | 120.5 \[120.5, 120.5\] | 10.1 \[10.1, 10.1\] | 6.1 \[5.9, 6.1\] | 139.7 \[139.6, 139.8\] |
| epydemic | 10000 | 117.3 \[117.3, 117.3\] | 10.1 \[10.1, 10.1\] | 16.0 \[16.0, 16.0\] | 146.6 \[146.6, 146.6\] |
| covasim | 10000 | 225.3 \[225.3, 225.3\] | 1.8 \[1.8, 1.8\] | 5.4 \[5.2, 5.5\] | 235.7 \[235.7, 235.8\] |
| starsim | 10000 | 261.1 \[261.1, 261.1\] | 0.0 \[0.0, 0.0\] | 8.1 \[8.0, 8.3\] | 272.3 \[272.2, 272.4\] |
| EpiModel | 10000 | 231.3 \[231.3, 231.3\] | 16.2 \[16.2, 16.2\] | 41.8 \[29.9, 42.2\] | 290.9 \[289.7, 296.3\] |
| Agents.jl | 10000 | 507.3 \[507.0, 507.5\] | 0.5 \[0.5, 0.5\] | 0.4 \[0.4, 0.5\] | 532.3 \[531.5, 534.4\] |
| ixa | 100000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 32.1 \[32.1, 32.1\] | 42.2 \[42.2, 42.2\] |
| epiworld | 100000 | 2.9 \[2.9, 2.9\] | 23.5 \[23.5, 23.5\] | 12.2 \[12.1, 12.2\] | 43.5 \[43.5, 43.6\] |
| epiworldR | 100000 | 72.1 \[72.1, 72.1\] | 23.6 \[23.6, 23.6\] | 0.0 \[0.0, 0.0\] | 113.4 \[113.4, 113.4\] |
| epiworldpy | 100000 | 46.7 \[46.7, 46.7\] | 39.6 \[39.6, 39.6\] | 0.9 \[0.9, 1.0\] | 132.1 \[132.1, 132.1\] |
| individual | 100000 | 74.8 \[74.8, 74.8\] | 6.2 \[6.2, 6.2\] | 49.6 \[49.2, 49.6\] | 142.0 \[141.6, 142.0\] |
| ABM\* | 100000 | 74.4 \[74.4, 74.4\] | 6.2 \[6.2, 6.2\] | 95.6 \[95.5, 95.7\] | 191.1 \[191.0, 191.2\] |
| covasim | 100000 | 225.3 \[225.3, 225.3\] | 3.2 \[3.2, 3.5\] | 41.9 \[41.9, 42.2\] | 281.0 \[280.8, 281.2\] |
| EoN\* | 100000 | 120.5 \[120.5, 120.5\] | 126.4 \[125.9, 126.4\] | 56.8 \[56.8, 56.9\] | 312.9 \[312.9, 312.9\] |
| starsim | 100000 | 261.1 \[261.1, 261.1\] | 0.0 \[0.0, 0.0\] | 69.3 \[69.1, 69.7\] | 340.6 \[340.5, 340.9\] |
| epydemic | 100000 | 117.3 \[117.3, 117.3\] | 126.2 \[126.2, 126.2\] | 160.5 \[160.5, 160.5\] | 413.4 \[413.4, 413.4\] |
| EpiModel | 100000 | 231.3 \[231.3, 231.3\] | 137.1 \[137.1, 137.1\] | 98.1 \[98.1, 98.1\] | 477.4 \[477.4, 477.4\] |
| Agents.jl | 100000 | 507.1 \[506.7, 507.3\] | 1.4 \[1.4, 1.8\] | 2.2 \[2.1, 2.4\] | 585.6 \[584.4, 588.0\] |
| FRED | 100000 |  |  |  | 598.5 \[598.5, 598.6\] |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

### Epidemiological sanity checks

| Engine | Agents | Median final attack rate | Median peak hospitalized | Median share protected |
|:---|---:|---:|---:|---:|
| ABM\* | 10000 | 0.114 | 9.0 | 0.24 |
| Agents.jl | 10000 | 0.118 | 8.0 | 0.24 |
| covasim | 10000 | 0.105 | 11.0 | 0.24 |
| EoN\* | 10000 | 0.113 | 10.0 | 0.24 |
| EpiModel | 10000 | 0.104 | 9.0 | 0.24 |
| epiworld | 10000 | 0.119 | 8.0 | 0.24 |
| epiworldpy | 10000 | 0.117 | 8.0 | 0.24 |
| epiworldR | 10000 | 0.115 | 9.0 | 0.24 |
| epydemic | 10000 | 0.121 | 11.0 | 0.24 |
| FRED | 10000 | 0.115 | 10.0 | 0.24 |
| individual | 10000 | 0.116 | 9.0 | 0.24 |
| ixa | 10000 | 0.119 | 8.0 | 0.24 |
| starsim | 10000 | 0.119 | 9.0 | 0.24 |
| ABM\* | 100000 | 0.013 | 9.5 | 0.24 |
| Agents.jl | 100000 | 0.014 | 9.0 | 0.24 |
| covasim | 100000 | 0.012 | 11.0 | 0.24 |
| EoN\* | 100000 | 0.013 | 10.0 | 0.24 |
| EpiModel | 100000 | 0.012 | 9.0 | 0.24 |
| epiworld | 100000 | 0.014 | 9.0 | 0.24 |
| epiworldpy | 100000 | 0.014 | 9.0 | 0.24 |
| epiworldR | 100000 | 0.014 | 9.0 | 0.24 |
| epydemic | 100000 | 0.014 | 11.0 | 0.24 |
| FRED | 100000 | 0.012 | 10.0 | 0.24 |
| individual | 100000 | 0.013 | 10.0 | 0.24 |
| ixa | 100000 | 0.014 | 9.0 | 0.24 |
| starsim | 100000 | 0.014 | 10.0 | 0.24 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

Share of runs whose outputs agree with their own final counts (every
agent who left S is a seed or a transmission target, every transmission
is an S → E transition, every recovered agent came from I or H):

| Engine | Agents | Median transmissions | Share of runs: tree matches counts | Share of runs: matrix matches counts |
|:---|---:|---:|---:|---:|
| epiworld | 10000 | 1090 | 1 | 1 |
| epiworldR | 10000 | 1050 | 1 | 1 |
| epiworldpy | 10000 | 1067 | 1 | 1 |
| covasim | 10000 | 947 | 1 | 1 |
| starsim | 10000 | 1092 | 1 | 1 |
| EoN\* | 10000 | 1034 | 1 | 1 |
| epydemic | 10000 | 1108 | 1 | 1 |
| ixa | 10000 | 1088 | 1 | 1 |
| individual | 10000 | 1060 | 1 | 1 |
| ABM\* | 10000 | 1043 | 1 | 1 |
| EpiModel | 10000 | 944 | 1 | 1 |
| FRED | 10000 | 1046 | 1 | 1 |
| Agents.jl | 10000 | 1081 | 1 | 1 |
| epiworld | 100000 | 1288 | 1 | 1 |
| epiworldR | 100000 | 1293 | 1 | 1 |
| epiworldpy | 100000 | 1264 | 1 | 1 |
| covasim | 100000 | 1084 | 1 | 1 |
| starsim | 100000 | 1262 | 1 | 1 |
| EoN\* | 100000 | 1188 | 1 | 1 |
| epydemic | 100000 | 1281 | 1 | 1 |
| ixa | 100000 | 1269 | 1 | 1 |
| individual | 100000 | 1191 | 1 | 1 |
| ABM\* | 100000 | 1160 | 1 | 1 |
| EpiModel | 100000 | 1062 | 1 | 1 |
| FRED | 100000 | 1115 | 1 | 1 |
| Agents.jl | 100000 | 1250 | 1 | 1 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

Daily series, as medians across replicates. Engines that seed their
initial cases exposed (the epiworld family, Covasim, and Starsim) start
their incidence a few days later.

![](README_files/figure-commonmark/daily-plot-1.png)

### Model lines

Counted as described in [Implementation
size](../docs/methods.md#implementation-size); the last column is what
the outputs added to scenario 01. Code that only summarizes the outputs
for the result record is not counted.

| Engine | Language | Files | Lines, scenario 01 | Lines, scenario 02 | Added since scenario 01 |
|:---|:---|---:|---:|---:|---:|
| epiworldpy | Python | 1 | 39 | 53 | 14 |
| epiworld | C++ | 1 | 34 | 57 | 23 |
| ABM\* | R | 1 | 45 | 59 | 14 |
| epiworldR | R | 1 | 62 | 66 | 4 |
| EoN\* | Python | 1 | 45 | 77 | 32 |
| individual | R | 1 | 37 | 80 | 43 |
| EpiModel | R | 1 | 73 | 99 | 26 |
| covasim | Python | 1 | 76 | 110 | 34 |
| Agents.jl | Julia | 1 | 83 | 115 | 32 |
| epydemic | Python | 1 | 89 | 122 | 33 |
| starsim | Python | 1 | 95 | 137 | 42 |
| FRED | Python | 1 | 123 | 165 | 42 |
| ixa | Rust | 3 | 168 | 220 | 52 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

## Notes

**Extraction is small for every engine but epiworldR, EpiModel, and
FRED.** FRED’s is parsing its health records. EpiModel’s is
`get_transmat()`, which keeps one infector per newly infected agent by
grouping the transmission matrix with dplyr and sampling within each
group. That costs about 45 µs per transmission, about 45 ms for the
roughly 1,000 transmissions of a replicate here, and grows with the
outbreak. Almost all of epiworldR’s is the documented
`plot(get_reproductive_number(model), plot = FALSE)`, which computes a
mean, a standard deviation, and two quantiles for every day and builds a
data frame for each: about 25 ms for 100 days however small the
outbreak, against 0.4 ms for a plain `tapply()` on the same data and
about 1 ms for `get_reproductive_number()` itself. The runner keeps the
documented call, as an epiworldR user would write it; a vectorized
summary in epiworldR would remove nearly all of the package’s extraction
time.
