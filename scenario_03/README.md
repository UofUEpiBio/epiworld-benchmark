# Scenario 03: SEIRH baseline at 1,000,000 agents

2026-09-28

- [Model](#model)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
  - [The epiworld family](#the-epiworld-family)
  - [Growth with the population](#growth-with-the-population)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
- [Interpretation](#interpretation)
  - [How time grows with the
    population](#how-time-grows-with-the-population)
  - [Building once](#building-once)
  - [Why this scenario enforces one replicate at a
    time](#why-this-scenario-enforces-one-replicate-at-a-time)
- [Recorded versions](#recorded-versions)

[Back to the project overview](../README.md) · [Scenario 00
report](../scenario_00/README.md)

[Scenario 00](../scenario_00/README.md) at 1,000,000 agents. The model,
the parameters, the network’s mean degree and rewiring probability, the
100 seed cases, and every runner are those of scenario 00; only the
population is ten times its largest size. Each engine runs 20 replicates
rather than 100, because the slowest ones take several seconds per
replicate here. Like every scenario, it runs one replicate at a time;
its `[design]` table in [`scenario.toml`](scenario.toml) sets the size
and replicate count, and enforces the one-at-a-time rule ([see
why](#why-this-scenario-enforces-one-replicate-at-a-time)).

The question this scenario asks is how each engine’s time grows with the
population when the outbreak does not. With 100 seed cases and the same
per-contact risk, the epidemic stays about the same size in absolute
terms: about 6,100 agents are infected at 100,000 agents and 6,600 at
1,000,000 (median attack rates 0.061 and 0.0066). An engine whose work
follows the outbreak should barely slow down. An engine that touches
every agent every day should slow down about tenfold.

## Model

See [scenario 00](../scenario_00/README.md#model) for the model, its
parameters, the timing conventions, and each engine’s implementation.
The runners in [`runners/`](runners) are copies of scenario 00’s.

The transmission multipliers were not recalibrated at this size. Each
engine reuses its 100,000-agent factor from scenario 00, which barely
moved between 10,000 and 100,000 agents. The attack rates below show
that the engines stay aligned.

## Results

> [!TIP]
>
> The complete 180-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 1                                                     |
| Latest run failures | 0                                                     |

| Engine     |  Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|--------:|-----:|----------------------:|-------:|-------:|
| epiworldR  | 1000000 |   20 |                 0.033 |  0.032 |  0.036 |
| epiworld   | 1000000 |   20 |                 0.036 |  0.033 |  0.039 |
| epiworldpy | 1000000 |   20 |                 0.040 |  0.035 |  0.048 |
| individual | 1000000 |   20 |                 0.316 |  0.280 |  0.352 |
| ixa        | 1000000 |   20 |                 0.460 |  0.408 |  0.473 |
| EoN        | 1000000 |   20 |                 2.606 |  2.526 |  2.832 |
| covasim    | 1000000 |   20 |                 3.544 |  3.319 |  3.644 |
| starsim    | 1000000 |   20 |                11.884 | 11.416 | 12.909 |
| epydemic   | 1000000 |   20 |                16.569 | 15.535 | 18.218 |

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by replicate seed. Values above one mean that
epiworldR completed the simulation call faster.

| Engine     |  Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|--------:|------------------------:|-------:|-------:|
| epiworld   | 1000000 |                    1.05 |   0.97 |   1.14 |
| epiworldpy | 1000000 |                    1.18 |   1.07 |   1.40 |
| covasim    | 1000000 |                  101.60 |  95.85 | 109.08 |
| starsim    | 1000000 |                  362.93 | 348.13 | 374.31 |
| EoN        | 1000000 |                   79.76 |  75.54 |  87.74 |
| epydemic   | 1000000 |                  490.22 | 461.53 | 555.34 |
| ixa        | 1000000 |                   13.81 |  11.92 |  14.79 |
| individual | 1000000 |                    9.70 |   8.32 |  10.87 |

### Time to a first result

The simulation time above covers what an engine has to redo for every
replicate on the same network (see [what is
measured](../README.md#run-time)). *Build* is the rest of the setup:
turning the edge list into the engine’s population and network, which a
user does once per network. *Build + simulate* is the time to a first
result once the edge list is in memory. Reading the edge file is shown
but left out of it, because its cost depends on each language’s file
parsing rather than on the engine.

| Engine     |  Agents | Read edges (s) | Build (s) | Simulate (s) | Build + simulate (s) |
|:-----------|--------:|---------------:|----------:|-------------:|---------------------:|
| epiworld   | 1000000 |          0.510 |     0.144 |        0.036 |                0.180 |
| epiworldR  | 1000000 |          0.857 |     0.194 |        0.033 |                0.226 |
| ixa        | 1000000 |          0.306 |     0.003 |        0.460 |                0.464 |
| epiworldpy | 1000000 |          1.784 |     0.482 |        0.040 |                0.540 |
| individual | 1000000 |          0.994 |     0.320 |        0.316 |                0.638 |
| covasim    | 1000000 |          1.798 |     0.082 |        3.544 |                3.635 |
| EoN        | 1000000 |          1.697 |     4.290 |        2.606 |                6.829 |
| starsim    | 1000000 |          1.680 |     0.002 |       11.884 |               11.886 |
| epydemic   | 1000000 |          1.772 |     4.339 |       16.569 |               20.774 |

### The epiworld family

| Engine     |  Agents | Median time / epiworld |   Q1 |   Q3 |
|:-----------|--------:|-----------------------:|-----:|-----:|
| epiworldR  | 1000000 |                   0.95 | 0.87 | 1.03 |
| epiworldpy | 1000000 |                   1.08 | 1.01 | 1.21 |

### Growth with the population

Median simulation time at each size, the first two from scenario 00. The
growth column divides the time at 1,000,000 agents by the time at
100,000; the population grew tenfold, the outbreak hardly at all. The
last column is the time to build the model from the edge list, which is
not part of the simulation time (see [Building once](#building-once)).

| Engine | 10,000 agents (s) | 100,000 agents (s) | 1,000,000 agents (s) | Time at 1,000,000 / at 100,000 | Median build at 1,000,000 (s) |
|:---|---:|---:|---:|---:|---:|
| epiworldR | 0.009 | 0.015 | 0.033 | 2.2 | 0.194 |
| epiworld | 0.009 | 0.015 | 0.036 | 2.4 | 0.144 |
| epiworldpy | 0.008 | 0.014 | 0.040 | 2.8 | 0.482 |
| individual | 0.026 | 0.060 | 0.316 | 5.3 | 0.320 |
| ixa | 0.007 | 0.038 | 0.460 | 12.0 | 0.003 |
| EoN | 0.079 | 0.306 | 2.606 | 8.5 | 4.290 |
| covasim | 0.065 | 0.329 | 3.544 | 10.8 | 0.082 |
| starsim | 0.174 | 0.938 | 11.884 | 12.7 | 0.002 |
| epydemic | 0.509 | 1.854 | 16.569 | 8.9 | 4.339 |

### Epidemiological sanity checks

| Engine     |  Agents | Median final attack rate | Median peak hospitalized |
|:-----------|--------:|-------------------------:|-------------------------:|
| covasim    | 1000000 |                    0.007 |                     38.0 |
| EoN        | 1000000 |                    0.007 |                     41.0 |
| epiworld   | 1000000 |                    0.006 |                     32.0 |
| epiworldpy | 1000000 |                    0.006 |                     32.0 |
| epiworldR  | 1000000 |                    0.006 |                     32.0 |
| epydemic   | 1000000 |                    0.007 |                     43.5 |
| individual | 1000000 |                    0.006 |                     34.0 |
| ixa        | 1000000 |                    0.006 |                     30.5 |
| starsim    | 1000000 |                    0.007 |                     40.0 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

## Interpretation

### How time grows with the population

- **The epiworld runners** grow by 2.4 (C++), 2.2 (epiworldR), and 2.8
  (epiworldpy) times. Their daily work follows the outbreak, which
  barely grew. What does grow is `reset()` at the start of `run()`,
  which re-initializes every agent; at this size it is nearly half of
  the run.
- **individual** grows by 5.3 times. Its infection process visits the
  infectious agents’ neighbours, but it also tabulates contacts and
  intersects bitsets over the whole population every day.
- **ixa** grows by 12.0 times. Its `execute()` follows the outbreak, but
  every replicate first rebuilds the context (a million entities, five
  million edges, and the index), which grows with the population and is
  now most of its time.
- **EoN** (8.5 times) and **epydemic** (8.9) follow the outbreak in
  their event handling, but set up state for every node of the network,
  in Python, at the start of each run.
- **Covasim** (10.8 times) and **Starsim** (12.7) update every agent’s
  arrays every day, and Starsim also draws transmission on every edge,
  so they grow about as fast as the population.

At 1,000,000 agents, epydemic takes about 17 seconds per replicate and
Starsim 12, against 0.036 for the C++ runner and 0.46 for ixa.
[analysis.md](../analysis.md) measures where epiworld’s and ixa’s time
goes.

### Building once

Building the model from the edge list, which a user does once per
network, takes the epiworld runners 0.14 (C++), 0.19 (epiworldR), and
0.48 (epiworldpy) seconds: several times longer than a 100-day run, but
paid once however many replicates follow. EoN spends 4.3 seconds and
epydemic 4.3 building their NetworkX graphs, which they also reuse. ixa
and Starsim have almost nothing to build once, because they rebuild
their whole model for every replicate, and that time is in their
simulation time above.

### Why this scenario enforces one replicate at a time

In a first run with four workers, the scheduler ran four replicates of
the same engine at once, and at this size they competed for memory:
epydemic took 57 to 67 seconds per replicate, against about 17 run
alone, and epiworldpy and EoN slowed too. The `workers = 1` setting in
[`scenario.toml`](scenario.toml) makes this scenario run sequentially
whatever `N_THREADS` says. The other scenarios also run sequentially, by
default, so the growth factors above compare times collected the same
way.

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
