# Scenario 01: SEIRH with an all-or-nothing vaccine

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

[Scenario 00](../scenario_00/README.md) plus a vaccine given before
transmission starts. The disease, network, seeds, run length, and
calibration factors are scenario 00’s, so for any engine, size, and
replicate the two scenarios differ only in the vaccine; the
[results](../docs/results.md#cost-of-added-features) compare them
replicate by replicate.

## Model

| Parameter                     | Value |
|:------------------------------|------:|
| `hospital_days`               |     7 |
| `hospitalization_probability` |  0.05 |
| `infectious_days`             |     7 |
| `initial_infected`            |   100 |
| `latent_days`                 |     4 |
| `target_r0`                   |     2 |
| `vaccine_coverage`            |   0.3 |
| `vaccine_efficacy`            |   0.8 |

- Before the first day, `round(vaccine_coverage * n)` agents are chosen
  uniformly at random and vaccinated, independently of the seed cases.
- The vaccine is **all-or-nothing**: each vaccinee is protected with
  probability `vaccine_efficacy`, and a protected agent can never be
  infected. An unprotected vaccinee behaves like an unvaccinated agent,
  and a seed case that is also vaccinated stays infected. About 24% of
  agents end up protected, so the effective reproduction number falls
  from about 2 to about 1.5.
- Distributing the vaccine is timed as simulation, as seeding is
  ([Timing](../docs/methods.md#timing)).
- Records add `vaccinated` and `vaccine_protected`. Protected agents who
  are never infected count as susceptible, so the attack rate is still
  `1 - final_susceptible / n`.

## Engine notes

Each engine expresses the vaccine with its own tools:

- **The epiworld family**: a tool with `susceptibility_reduction = 1`.
  Each runner draws the number of protected agents (one binomial draw,
  with its own language’s random numbers) and lets epiworld place the
  tool at random, so the three runners’ epidemics now differ replicate
  by replicate, though not in distribution. epiworld’s `ToolVaccine`
  would decide protection only at first exposure, so the number
  protected would not be observable.
- **Covasim**: two day-0 `cv.simple_vaccine` interventions targeted with
  `subtarget`, one setting `rel_sus = 0` for the protected agents; a
  single one would be leaky.
- **Starsim**: its all-or-nothing `ss.simple_vx(leaky=False)`, delivered
  by a day-0 `ss.campaign_vx`. The runner seeds NumPy’s global
  generator, which `simple_vx` draws from.
- **EoN** and **epydemic**: a `V` status or compartment with no
  transitions.
- **ixa**: a `VaccineStatus` property set in a plan at time 0, which the
  daily step consults. **Agents.jl**: a `protected` field the daily step
  consults.
- **individual**: a `Bitset` of protected agents that the infection
  process leaves out.
- **FRED**: a second condition, `VAX`, written as in FRED’s own vaccine
  example. An import on day 0 moves the vaccinees into it, and
  `Protected` sets their susceptibility to zero.

## Results

> [!TIP]
>
> The complete 2,400-run design for this scenario is available.

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
| Run dates | 2026-09-30 to 2026-09-30 |
| Latest run | 2200 executed, 0 cached, 0 failed |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| epiworldR  |  10000 |  100 |                 0.002 |  0.002 |  0.002 |
| epiworldpy |  10000 |  100 |                 0.002 |  0.002 |  0.002 |
| epiworld   |  10000 |  100 |                 0.002 |  0.002 |  0.002 |
| ixa        |  10000 |  100 |                 0.004 |  0.004 |  0.004 |
| Agents.jl  |  10000 |  100 |                 0.020 |  0.020 |  0.020 |
| individual |  10000 |  100 |                 0.021 |  0.021 |  0.022 |
| EoN        |  10000 |  100 |                 0.034 |  0.033 |  0.036 |
| FRED       |  10000 |  100 |                 0.054 |  0.053 |  0.056 |
| covasim    |  10000 |  100 |                 0.062 |  0.061 |  0.064 |
| ABM        |  10000 |  100 |                 0.082 |  0.077 |  0.087 |
| starsim    |  10000 |  100 |                 0.169 |  0.167 |  0.178 |
| epydemic   |  10000 |  100 |                 0.203 |  0.195 |  0.215 |
| epiworldpy | 100000 |  100 |                 0.006 |  0.006 |  0.006 |
| epiworldR  | 100000 |  100 |                 0.007 |  0.006 |  0.007 |
| epiworld   | 100000 |  100 |                 0.007 |  0.007 |  0.007 |
| ixa        | 100000 |  100 |                 0.029 |  0.028 |  0.031 |
| Agents.jl  | 100000 |  100 |                 0.037 |  0.036 |  0.038 |
| individual | 100000 |  100 |                 0.049 |  0.048 |  0.050 |
| EoN        | 100000 |  100 |                 0.211 |  0.208 |  0.219 |
| covasim    | 100000 |  100 |                 0.334 |  0.333 |  0.336 |
| FRED       | 100000 |  100 |                 0.587 |  0.580 |  0.595 |
| ABM        | 100000 |  100 |                 0.868 |  0.785 |  0.971 |
| starsim    | 100000 |  100 |                 0.908 |  0.831 |  0.931 |
| epydemic   | 100000 |  100 |                 1.075 |  1.060 |  1.087 |

### Simulation time

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|-------:|------------------------:|-------:|-------:|
| epiworld   |  10000 |                    1.06 |   0.91 |   1.12 |
| epiworldpy |  10000 |                    0.97 |   0.82 |   1.10 |
| covasim    |  10000 |                   30.72 |  30.04 |  32.23 |
| starsim    |  10000 |                   83.86 |  82.95 |  88.40 |
| EoN        |  10000 |                   17.00 |  15.29 |  18.01 |
| epydemic   |  10000 |                  100.92 |  89.17 | 107.36 |
| ixa        |  10000 |                    2.12 |   1.94 |   2.21 |
| individual |  10000 |                   10.50 |  10.00 |  11.00 |
| ABM        |  10000 |                   39.50 |  36.00 |  42.62 |
| FRED       |  10000 |                   26.90 |  25.98 |  27.88 |
| Agents.jl  |  10000 |                    9.84 |   9.74 |   9.98 |
| epiworld   | 100000 |                    1.04 |   0.98 |   1.12 |
| epiworldpy | 100000 |                    0.87 |   0.81 |   0.96 |
| covasim    | 100000 |                   47.79 |  47.51 |  55.39 |
| starsim    | 100000 |                  129.73 | 116.46 | 139.36 |
| EoN        | 100000 |                   30.33 |  29.54 |  34.50 |
| epydemic   | 100000 |                  154.54 | 151.14 | 175.77 |
| ixa        | 100000 |                    4.21 |   4.04 |   4.74 |
| individual | 100000 |                    7.14 |   6.86 |   8.00 |
| ABM        | 100000 |                  128.86 | 113.02 | 156.79 |
| FRED       | 100000 |                   84.57 |  82.93 |  96.26 |
| Agents.jl  | 100000 |                    5.41 |   5.16 |   6.20 |

### Time to a first result

[Timing](../docs/methods.md#timing) defines build and simulate.

| Engine     | Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|-------:|---------------:|----------:|-------------:|---------------------:|
| epiworld   |  10000 |          0.005 |     0.001 |        0.002 |                0.003 |
| ixa        |  10000 |          0.003 |     0.000 |        0.004 |                0.004 |
| epiworldpy |  10000 |          0.016 |     0.007 |        0.002 |                0.009 |
| epiworldR  |  10000 |          0.007 |     0.016 |        0.002 |                0.018 |
| individual |  10000 |          0.007 |     0.018 |        0.021 |                0.039 |
| Agents.jl  |  10000 |          0.175 |     0.020 |        0.020 |                0.040 |
| EoN        |  10000 |          0.016 |     0.028 |        0.034 |                0.063 |
| covasim    |  10000 |          0.016 |     0.038 |        0.062 |                0.100 |
| ABM        |  10000 |          0.007 |     0.023 |        0.082 |                0.105 |
| starsim    |  10000 |          0.016 |     0.000 |        0.169 |                0.170 |
| FRED       |  10000 |          0.241 |     0.129 |        0.054 |                0.184 |
| epydemic   |  10000 |          0.015 |     0.017 |        0.203 |                0.221 |
| epiworld   | 100000 |          0.048 |     0.011 |        0.007 |                0.018 |
| ixa        | 100000 |          0.028 |     0.000 |        0.029 |                0.029 |
| epiworldR  | 100000 |          0.071 |     0.031 |        0.007 |                0.038 |
| epiworldpy | 100000 |          0.151 |     0.040 |        0.006 |                0.046 |
| Agents.jl  | 100000 |          0.276 |     0.037 |        0.037 |                0.075 |
| individual | 100000 |          0.068 |     0.030 |        0.049 |                0.079 |
| covasim    | 100000 |          0.154 |     0.039 |        0.334 |                0.373 |
| EoN        | 100000 |          0.157 |     0.243 |        0.211 |                0.457 |
| starsim    | 100000 |          0.151 |     0.000 |        0.908 |                0.908 |
| ABM        | 100000 |          0.076 |     0.043 |        0.868 |                0.923 |
| epydemic   | 100000 |          0.146 |     0.278 |        1.075 |                1.354 |
| FRED       | 100000 |          2.415 |     0.821 |        0.587 |                1.410 |

### Memory

Median \[Q1, Q3\] in MiB ([Memory](../docs/methods.md#memory)).

| Engine | Agents | Baseline (MiB) | Model footprint (MiB) | Simulation memory (MiB) | Overall peak (MiB) |
|:---|:---|---:|---:|---:|---:|
| ixa | 10000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 3.1 \[3.1, 3.1\] | 6.5 \[6.5, 6.5\] |
| epiworld | 10000 | 2.9 \[2.9, 2.9\] | 2.5 \[2.5, 2.5\] | 1.8 \[1.8, 1.8\] | 8.2 \[8.2, 8.2\] |
| epiworldpy | 10000 | 46.7 \[46.7, 46.7\] | 7.0 \[7.0, 7.0\] | 0.5 \[0.5, 0.6\] | 59.3 \[59.3, 59.3\] |
| FRED | 10000 |  |  |  | 70.1 \[70.0, 70.1\] |
| epiworldR | 10000 | 72.1 \[72.1, 72.1\] | 2.8 \[2.8, 2.8\] | 0.0 \[0.0, 0.0\] | 75.7 \[75.6, 75.7\] |
| individual | 10000 | 74.8 \[74.8, 74.8\] | 1.5 \[1.5, 1.5\] | 9.6 \[9.4, 9.9\] | 89.3 \[89.0, 89.4\] |
| ABM | 10000 | 74.4 \[74.4, 74.4\] | 1.9 \[1.9, 1.9\] | 15.1 \[14.9, 15.2\] | 92.0 \[91.8, 92.1\] |
| EoN | 10000 | 120.3 \[120.3, 120.4\] | 10.3 \[10.1, 10.3\] | 2.2 \[2.1, 2.2\] | 135.7 \[135.6, 135.8\] |
| epydemic | 10000 | 117.2 \[117.2, 117.2\] | 9.9 \[9.9, 10.1\] | 15.6 \[15.6, 15.6\] | 146.0 \[146.0, 146.0\] |
| covasim | 10000 | 225.2 \[225.1, 225.2\] | 1.8 \[1.8, 1.8\] | 5.4 \[5.2, 5.5\] | 235.6 \[235.5, 235.6\] |
| starsim | 10000 | 260.9 \[260.9, 260.9\] | 0.0 \[0.0, 0.0\] | 7.1 \[6.8, 7.1\] | 271.0 \[271.0, 271.0\] |
| Agents.jl | 10000 | 483.9 \[483.8, 484.2\] | 0.5 \[0.5, 0.5\] | 0.4 \[0.4, 0.4\] | 534.5 \[533.9, 535.1\] |
| ixa | 100000 | 2.2 \[2.2, 2.2\] | 0.0 \[0.0, 0.0\] | 32.1 \[32.1, 32.1\] | 42.4 \[42.4, 42.4\] |
| epiworld | 100000 | 2.9 \[2.9, 2.9\] | 23.5 \[23.5, 23.5\] | 12.2 \[12.1, 12.2\] | 43.2 \[43.1, 43.2\] |
| epiworldR | 100000 | 72.1 \[72.1, 72.1\] | 23.6 \[23.6, 23.6\] | 0.0 \[0.0, 0.0\] | 113.4 \[113.4, 113.4\] |
| individual | 100000 | 74.8 \[74.8, 74.8\] | 6.2 \[6.2, 6.2\] | 34.6 \[27.4, 34.9\] | 127.0 \[119.8, 127.3\] |
| epiworldpy | 100000 | 46.7 \[46.7, 46.7\] | 38.7 \[38.7, 39.6\] | 0.9 \[0.9, 1.0\] | 132.0 \[132.0, 132.0\] |
| ABM | 100000 | 74.4 \[74.4, 74.4\] | 6.2 \[6.2, 6.2\] | 96.2 \[96.2, 96.3\] | 190.9 \[190.9, 191.0\] |
| covasim | 100000 | 225.2 \[225.1, 225.2\] | 3.2 \[3.2, 3.5\] | 41.9 \[41.9, 42.2\] | 280.8 \[280.7, 281.1\] |
| EoN | 100000 | 120.3 \[120.3, 120.4\] | 125.9 \[125.9, 125.9\] | 30.4 \[30.4, 30.4\] | 286.3 \[286.3, 286.3\] |
| starsim | 100000 | 260.9 \[260.9, 260.9\] | 0.0 \[0.0, 0.0\] | 67.6 \[67.2, 67.7\] | 338.7 \[338.7, 338.7\] |
| epydemic | 100000 | 117.2 \[117.2, 117.2\] | 126.2 \[126.2, 126.4\] | 160.5 \[160.5, 160.5\] | 413.3 \[413.2, 413.3\] |
| Agents.jl | 100000 | 484.1 \[484.0, 484.4\] | 6.9 \[5.7, 8.5\] | 2.0 \[1.9, 2.1\] | 584.8 \[583.5, 586.1\] |
| FRED | 100000 |  |  |  | 597.4 \[597.4, 597.4\] |

### Epidemiological sanity checks

| Engine | Agents | Median final attack rate | Median peak hospitalized | Median share protected |
|:---|---:|---:|---:|---:|
| ABM | 10000 | 0.114 | 9.0 | 0.24 |
| Agents.jl | 10000 | 0.118 | 8.0 | 0.24 |
| covasim | 10000 | 0.105 | 11.0 | 0.24 |
| EoN | 10000 | 0.113 | 10.0 | 0.24 |
| epiworld | 10000 | 0.119 | 8.0 | 0.24 |
| epiworldpy | 10000 | 0.117 | 8.0 | 0.24 |
| epiworldR | 10000 | 0.115 | 9.0 | 0.24 |
| epydemic | 10000 | 0.121 | 11.0 | 0.24 |
| FRED | 10000 | 0.115 | 10.0 | 0.24 |
| individual | 10000 | 0.117 | 9.0 | 0.24 |
| ixa | 10000 | 0.119 | 8.0 | 0.24 |
| starsim | 10000 | 0.119 | 9.0 | 0.24 |
| ABM | 100000 | 0.013 | 9.5 | 0.24 |
| Agents.jl | 100000 | 0.014 | 9.0 | 0.24 |
| covasim | 100000 | 0.012 | 11.0 | 0.24 |
| EoN | 100000 | 0.013 | 10.0 | 0.24 |
| epiworld | 100000 | 0.014 | 9.0 | 0.24 |
| epiworldpy | 100000 | 0.014 | 9.0 | 0.24 |
| epiworldR | 100000 | 0.014 | 9.0 | 0.24 |
| epydemic | 100000 | 0.014 | 11.0 | 0.24 |
| FRED | 100000 | 0.012 | 10.0 | 0.24 |
| individual | 100000 | 0.013 | 10.0 | 0.24 |
| ixa | 100000 | 0.014 | 9.0 | 0.24 |
| starsim | 100000 | 0.014 | 10.0 | 0.24 |

### Model lines

Counted as described in [Implementation
size](../docs/methods.md#implementation-size); the last column is what
the vaccine added to scenario 00.

| Engine | Language | Files | Lines, scenario 00 | Lines, scenario 01 | Added since scenario 00 |
|:---|:---|---:|---:|---:|---:|
| epiworld | C++ | 1 | 26 | 34 | 8 |
| individual | R | 1 | 32 | 37 | 5 |
| epiworldpy | Python | 1 | 35 | 39 | 4 |
| ABM | R | 1 | 42 | 45 | 3 |
| EoN | Python | 1 | 38 | 45 | 7 |
| epiworldR | R | 1 | 47 | 62 | 15 |
| covasim | Python | 1 | 65 | 76 | 11 |
| Agents.jl | Julia | 1 | 68 | 83 | 15 |
| epydemic | Python | 1 | 71 | 89 | 18 |
| starsim | Python | 1 | 84 | 95 | 11 |
| FRED | Python | 1 | 92 | 123 | 31 |
| ixa | Rust | 3 | 138 | 168 | 30 |

## Notes

**The C++ runner is the slowest epiworld runner here at 100,000
agents**, although it runs the same library. Placing the vaccine
allocates about 24,000 tool objects inside `run()`. The C++ runner
starts with a small heap, which grows during those allocations; R and
Python grew a large heap while starting up. Telling glibc to keep a
large heap in reserve (`MALLOC_TOP_PAD_`) or to serve large blocks from
the heap (`MALLOC_MMAP_THRESHOLD_`) brought a median of 10.9 ms down to
about 9 ms. The runner is timed as a C++ user would write it.

**epiworldR’s `distribute_tool_to_set()`** was quadratic before 0.16.1:
every per-agent `add_tool()` cloned the tool with a copy of the whole
target list, so placing it on about 24,000 named agents took 4.6 seconds
at 100,000 agents, against 0.03 seconds for the approach the runner uses
(draw the count, let epiworld place the tool). Both approaches are fast
now.

**Agents.jl’s first runner** drew the vaccine in a loop at the script’s
top level, over Julia’s untyped globals, which doubled its simulation
time at 10,000 agents. Moving the loop into a function that the warm-up
compiles, as a Julia user would, removed the cost.
