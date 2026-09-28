# Scenario 02: SEIRH + vaccination + epidemiological outputs

2026-09-28

- [Model](#model)
  - [Outputs](#outputs)
  - [Timing](#timing)
  - [Result fields](#result-fields)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
  - [Extracting the outputs](#extracting-the-outputs)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
- [Cost of the outputs](#cost-of-the-outputs)
  - [Run time compared with scenario
    01](#run-time-compared-with-scenario-01)
  - [Code required to define the
    model](#code-required-to-define-the-model)
- [Interpretation](#interpretation)
  - [What recording the outputs
    costs](#what-recording-the-outputs-costs)
  - [Is bookkeeping what separates epiworld and
    ixa?](#is-bookkeeping-what-separates-epiworld-and-ixa)
  - [An implementation choice that mattered: epiworldR’s
    reproductive-number
    summary](#an-implementation-choice-that-mattered-epiworldrs-reproductive-number-summary)
- [Recorded versions](#recorded-versions)

[Back to the project overview](../README.md) · [Scenario 01
report](../scenario_01/README.md)

[Scenario 01](../scenario_01/README.md), unchanged, plus four outputs
that every engine has to produce from each run: the transmission tree,
daily incidence, the reproductive number, and the daily transition
matrix.

epiworld records all four during every run, in every scenario, whether
or not anyone reads them. The other engines record less by default, so
in scenarios 00 and 01 the epiworld family does bookkeeping that the
others skip. Here every engine has to deliver the same information, so
the comparison includes that bookkeeping for all of them.

The model, network, seeds, run length, and calibration are those of
scenario 01. For any engine, size, and replicate, the two scenarios run
the same epidemic, and this report compares them replicate by replicate.

## Model

### Outputs

Each runner produces the following from each run and holds them in
memory in the engine’s own data structures. Nothing is written to disk.

| Output | Definition |
|:---|:---|
| Transmission tree | The day, source, and target of every infection. Seed cases have no source. |
| Daily incidence | New infections (S → E) on each day from 1 to 100. |
| Reproductive number | epiworld’s definition: each case’s number of secondary infections, averaged over the cases infected on each day. Day 0 holds the seed cases. |
| Transition matrix | The number of agents moving between each pair of S, E, I, H, and R on each day. |

- The reproductive number is a *case* reproductive number: it is indexed
  by the day the case was infected, not the day it transmitted. Cases
  infected near day 100 have had little time to transmit, so the series
  falls toward zero at the end of the run in every engine.
- Where an engine’s own tools already give an output, the runner uses
  them. Where they do not, the runner builds it from what the engine
  does offer, the way a user of that engine would.
- Transitions on day 0 put the seed cases in place and are left out of
  the daily series and the totals below.

### Timing

`simulate_seconds` covers the simulation call only, as in scenarios 00
and 01. Any bookkeeping an engine does while it simulates, such as
recording transmissions or transitions, is part of that call and is
timed. Reading the outputs out of the engine afterwards is not: it is
reported on its own as `extract_seconds` and left out of
`simulate_seconds` and `setup_seconds`. As in scenario 01, distributing
the vaccine is timed as simulation in every runner.

This puts the line at the end of the run. An engine that records outputs
as it goes pays for them in the timed call; an engine that reconstructs
them afterwards (Covasim’s transition matrix, EoN’s matrix from its node
histories) does that part untimed. The extraction table below shows how
much that is.

### Result fields

Runners add these fields to each result record:

- `extract_seconds`: the time taken to extract the outputs after the
  run.
- `transmissions`: the number of infections with a source, which is the
  size of the transmission tree without the seed cases.
- `transitions_se`, `transitions_ei`, `transitions_ih`,
  `transitions_ir`, `transitions_hr`: the transition matrix summed over
  days 1 to 100.
- `daily_incidence` (days 1 to 100) and `reproductive_number` (days 0 to
  100, `null` for a day with no cases): the daily series. They stay in
  the cache; `results/daily.csv` holds their medians across replicates.

The five final compartments still sum to `n`, and protected agents who
are never infected are still counted as susceptible.

### Engine implementations

- **epiworldR**: nothing to add to the model. After the run,
  `get_transmissions()` returns the tree and
  `get_hist_transition_matrix()` the matrix.
  `plot_incidence(plot = FALSE)` and
  `plot(get_reproductive_number(), plot = FALSE)` return the daily
  series. Four lines.
- **epiworld** (C++): the same database, through `get_transmissions()`,
  `get_hist_transition_matrix()`, and `get_reproductive_number()`. The
  C++ API returns raw vectors and a map with one entry per case, so the
  runner sums the matrix’s S → E entries and averages the reproductive
  number by day itself.
- **epiworldpy**: the same getters, returning NumPy arrays and
  dictionaries. The runner aggregates them with NumPy.
- **Covasim**: Covasim always keeps an infection log with the source,
  target, and date of every infection, and reports daily new infections.
  The runner reads the log directly: `sim.make_transtree()` wraps the
  same log but also builds a per-case table, and it drops the cases
  infected by agent 0. Covasim has no transition matrix, but it dates
  each agent’s transitions (`date_exposed`, `date_infectious`,
  `date_severe`, `date_recovered`, some of them in the future), and the
  runner counts those that fall within the run.
- **Starsim**: adding its `ss.infection_log` analyzer makes the disease
  log the source and target of every infection, in a NetworkX graph, as
  it happens; daily new infections are a standard result. Like Covasim,
  Starsim has no transition matrix but keeps each agent’s transition
  times, some of them scheduled in the future, and the runner counts
  those that fall within the run.
- **EoN**: `return_full_data=True` makes `fast_simple_contagion` record
  each node’s status history and every transmission, and return them in
  a `Simulation_Investigation`. The runner builds the matrix from the
  node histories. EoN runs in continuous time, so events are binned into
  days: day *d* covers the interval (*d* − 1, *d*\].
- **epydemic**: no built-in record of individual events. The model’s
  event handlers append each transmission and each transition to lists,
  through a small `move()` helper that records a transition before
  changing the compartment.
- **ixa**: a data plugin holds the outputs. The daily step, which
  already knows the infectious contact behind each exposure, records the
  transmission when it applies it. A subscription to ixa’s
  `PropertyChangeEvent` for the disease status counts every transition.
  After the run, the runner derives incidence and the reproductive
  number from the plugin.
- **individual**: no built-in record of events or sources. The infection
  process picks each new case’s source uniformly among its infectious
  neighbours, as epiworld does, and records it with the day. A second
  process compares the agents in each state at the start of consecutive
  days to count the transitions. After the run, the runner derives the
  reproductive number from the tree.

The EoN, epydemic, Covasim, Starsim, ixa, and individual runners use
epiworld’s definition of the reproductive number. Each Python runner has
its own copy of a small helper that computes it from the tree, and that
helper counts as model code.

## Results

> [!TIP]
>
> The complete 1,800-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 4                                                     |
| Latest run failures | 0                                                     |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| epiworldpy |  10000 |  100 |                 0.005 |  0.004 |  0.006 |
| epiworld   |  10000 |  100 |                 0.005 |  0.004 |  0.005 |
| epiworldR  |  10000 |  100 |                 0.005 |  0.004 |  0.007 |
| ixa        |  10000 |  100 |                 0.005 |  0.005 |  0.006 |
| individual |  10000 |  100 |                 0.054 |  0.047 |  0.067 |
| covasim    |  10000 |  100 |                 0.071 |  0.067 |  0.075 |
| EoN        |  10000 |  100 |                 0.084 |  0.078 |  0.093 |
| starsim    |  10000 |  100 |                 0.265 |  0.215 |  0.296 |
| epydemic   |  10000 |  100 |                 0.281 |  0.251 |  0.307 |
| epiworldR  | 100000 |  100 |                 0.012 |  0.011 |  0.014 |
| epiworldpy | 100000 |  100 |                 0.012 |  0.011 |  0.014 |
| epiworld   | 100000 |  100 |                 0.015 |  0.014 |  0.016 |
| ixa        | 100000 |  100 |                 0.041 |  0.038 |  0.043 |
| individual | 100000 |  100 |                 0.101 |  0.097 |  0.106 |
| covasim    | 100000 |  100 |                 0.442 |  0.425 |  0.465 |
| EoN        | 100000 |  100 |                 0.580 |  0.560 |  0.607 |
| starsim    | 100000 |  100 |                 1.171 |  1.121 |  1.297 |
| epydemic   | 100000 |  100 |                 1.463 |  1.390 |  1.538 |

### Simulation time

The primary measure is wall-clock time inside the simulation call,
including any recording the engine does during it. The logarithmic scale
keeps fast and slow engines legible in one panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR finished faster.

| Engine     | Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|-------:|------------------------:|-------:|-------:|
| epiworld   |  10000 |                    0.93 |   0.69 |   1.14 |
| epiworldpy |  10000 |                    0.89 |   0.71 |   1.15 |
| covasim    |  10000 |                   13.79 |  11.00 |  16.92 |
| starsim    |  10000 |                   48.94 |  35.11 |  64.93 |
| EoN        |  10000 |                   16.72 |  12.98 |  19.86 |
| epydemic   |  10000 |                   54.74 |  41.86 |  66.28 |
| ixa        |  10000 |                    1.05 |   0.79 |   1.29 |
| individual |  10000 |                   10.70 |   7.62 |  13.60 |
| epiworld   | 100000 |                    1.24 |   1.06 |   1.42 |
| epiworldpy | 100000 |                    1.03 |   0.87 |   1.19 |
| covasim    | 100000 |                   37.67 |  31.66 |  41.85 |
| starsim    | 100000 |                   99.68 |  86.74 | 112.58 |
| EoN        | 100000 |                   49.39 |  40.80 |  55.30 |
| epydemic   | 100000 |                  123.53 | 104.61 | 137.87 |
| ixa        | 100000 |                    3.38 |   2.82 |   3.81 |
| individual | 100000 |                    8.61 |   7.33 |   9.54 |

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
| ixa        |  10000 |          0.006 |     0.000 |        0.005 |                0.005 |
| epiworld   |  10000 |          0.006 |     0.001 |        0.005 |                0.006 |
| epiworldpy |  10000 |          0.022 |     0.009 |        0.005 |                0.013 |
| epiworldR  |  10000 |          0.008 |     0.021 |        0.005 |                0.027 |
| individual |  10000 |          0.009 |     0.022 |        0.054 |                0.083 |
| covasim    |  10000 |          0.019 |     0.051 |        0.071 |                0.121 |
| EoN        |  10000 |          0.019 |     0.047 |        0.084 |                0.130 |
| starsim    |  10000 |          0.018 |     0.000 |        0.265 |                0.266 |
| epydemic   |  10000 |          0.018 |     0.020 |        0.281 |                0.303 |
| epiworld   | 100000 |          0.055 |     0.015 |        0.015 |                0.029 |
| ixa        | 100000 |          0.032 |     0.000 |        0.041 |                0.041 |
| epiworldR  | 100000 |          0.088 |     0.041 |        0.012 |                0.054 |
| epiworldpy | 100000 |          0.190 |     0.065 |        0.012 |                0.079 |
| individual | 100000 |          0.078 |     0.039 |        0.101 |                0.141 |
| covasim    | 100000 |          0.179 |     0.056 |        0.442 |                0.501 |
| EoN        | 100000 |          0.178 |     0.354 |        0.580 |                0.936 |
| starsim    | 100000 |          0.177 |     0.000 |        1.171 |                1.172 |
| epydemic   | 100000 |          0.178 |     0.426 |        1.463 |                1.900 |

### Extracting the outputs

The time taken to read the four outputs out of each engine after the
run. It is not part of the simulation times above.

| Engine | Agents | Median simulation (s) | Median extraction (s) | Median extraction / simulation |
|:---|---:|---:|---:|---:|
| epiworldpy | 10000 | 0.005 | 0.000 | 0.11 |
| epiworld | 10000 | 0.005 | 0.000 | 0.05 |
| epiworldR | 10000 | 0.005 | 0.034 | 7.00 |
| ixa | 10000 | 0.005 | 0.000 | 0.02 |
| individual | 10000 | 0.054 | 0.001 | 0.02 |
| covasim | 10000 | 0.071 | 0.000 | 0.00 |
| EoN | 10000 | 0.084 | 0.002 | 0.02 |
| starsim | 10000 | 0.265 | 0.001 | 0.00 |
| epydemic | 10000 | 0.281 | 0.000 | 0.00 |
| epiworldR | 100000 | 0.012 | 0.035 | 3.00 |
| epiworldpy | 100000 | 0.012 | 0.001 | 0.05 |
| epiworld | 100000 | 0.015 | 0.000 | 0.02 |
| ixa | 100000 | 0.041 | 0.000 | 0.00 |
| individual | 100000 | 0.101 | 0.002 | 0.02 |
| covasim | 100000 | 0.442 | 0.001 | 0.00 |
| EoN | 100000 | 0.580 | 0.008 | 0.01 |
| starsim | 100000 | 1.171 | 0.002 | 0.00 |
| epydemic | 100000 | 1.463 | 0.000 | 0.00 |

### Epidemiological sanity checks

| Engine | Agents | Median final attack rate | Median peak hospitalized | Median share protected |
|:---|---:|---:|---:|---:|
| covasim | 10000 | 0.105 | 11 | 0.24 |
| EoN | 10000 | 0.113 | 10 | 0.24 |
| epiworld | 10000 | 0.119 | 8 | 0.24 |
| epiworldpy | 10000 | 0.117 | 8 | 0.24 |
| epiworldR | 10000 | 0.115 | 9 | 0.24 |
| epydemic | 10000 | 0.121 | 11 | 0.24 |
| individual | 10000 | 0.116 | 9 | 0.24 |
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

The outputs have to agree with each engine’s own final counts. Every
agent who left S is either a seed case or the target of a transmission;
every transmission is an S → E transition; and every recovered agent
came from I or H. The table shows the share of runs in which these hold.

| Engine | Agents | Median transmissions | Share of runs: tree matches counts | Share of runs: matrix matches counts |
|:---|---:|---:|---:|---:|
| epiworld | 10000 | 1090 | 1 | 1 |
| epiworldR | 10000 | 1050 | 1 | 1 |
| epiworldpy | 10000 | 1067 | 1 | 1 |
| covasim | 10000 | 947 | 1 | 1 |
| starsim | 10000 | 1092 | 1 | 1 |
| EoN | 10000 | 1034 | 1 | 1 |
| epydemic | 10000 | 1108 | 1 | 1 |
| ixa | 10000 | 1088 | 1 | 1 |
| individual | 10000 | 1060 | 1 | 1 |
| epiworld | 100000 | 1288 | 1 | 1 |
| epiworldR | 100000 | 1293 | 1 | 1 |
| epiworldpy | 100000 | 1264 | 1 | 1 |
| covasim | 100000 | 1084 | 1 | 1 |
| starsim | 100000 | 1262 | 1 | 1 |
| EoN | 100000 | 1188 | 1 | 1 |
| epydemic | 100000 | 1281 | 1 | 1 |
| ixa | 100000 | 1269 | 1 | 1 |
| individual | 100000 | 1191 | 1 | 1 |

The daily series, as medians across replicates. The epiworld family
seeds its initial cases as exposed, and so do Covasim and Starsim, so
their incidence starts a few days later than that of the engines that
seed them as infected.

![](README_files/figure-commonmark/daily-plot-1.png)

## Cost of the outputs

### Run time compared with scenario 01

Each replicate is paired with the scenario 01 replicate that has the
same engine, population size, and seed. Both scenarios run the same
epidemic, so a ratio above one is the cost of recording the outputs
during the run.

| Engine | Agents | Median scenario 01 (s) | Median scenario 02 (s) | Median time / scenario 01 | Q1 | Q3 |
|:---|---:|---:|---:|---:|---:|---:|
| epiworld | 10000 | 0.006 | 0.005 | 0.78 | 0.60 | 0.95 |
| covasim | 10000 | 0.085 | 0.071 | 0.84 | 0.74 | 0.95 |
| epydemic | 10000 | 0.305 | 0.281 | 0.96 | 0.82 | 1.03 |
| epiworldpy | 10000 | 0.004 | 0.005 | 0.98 | 0.90 | 1.12 |
| epiworldR | 10000 | 0.005 | 0.005 | 1.00 | 0.79 | 1.20 |
| ixa | 10000 | 0.005 | 0.005 | 1.00 | 0.89 | 1.11 |
| starsim | 10000 | 0.240 | 0.265 | 1.03 | 0.91 | 1.34 |
| individual | 10000 | 0.033 | 0.054 | 1.61 | 1.23 | 2.04 |
| EoN | 10000 | 0.044 | 0.084 | 1.97 | 1.43 | 2.24 |
| ixa | 100000 | 0.057 | 0.041 | 0.70 | 0.60 | 0.82 |
| starsim | 100000 | 1.495 | 1.171 | 0.79 | 0.65 | 1.00 |
| epiworld | 100000 | 0.015 | 0.015 | 0.95 | 0.79 | 1.06 |
| epiworldR | 100000 | 0.012 | 0.012 | 1.00 | 0.92 | 1.33 |
| epiworldpy | 100000 | 0.012 | 0.012 | 1.03 | 0.85 | 1.23 |
| epydemic | 100000 | 1.465 | 1.463 | 1.04 | 0.90 | 1.09 |
| covasim | 100000 | 0.417 | 0.442 | 1.08 | 0.95 | 1.16 |
| individual | 100000 | 0.080 | 0.101 | 1.30 | 1.09 | 1.48 |
| EoN | 100000 | 0.309 | 0.580 | 1.85 | 1.71 | 2.01 |

### Code required to define the model

How much code each engine needs for this scenario, and how much the
outputs added to scenario 01. The counting rules are described in the
[project overview](../README.md#measuring-implementation-effort), and
the counted regions are listed in
[`code_regions.yml`](code_regions.yml). Code that only summarizes the
outputs for the result record is not counted.

| Engine | Language | Files | Lines, scenario 01 | Lines, scenario 02 | Added since scenario 01 |
|:---|:---|---:|---:|---:|---:|
| epiworldpy | Python | 1 | 38 | 52 | 14 |
| epiworld | C++ | 1 | 34 | 57 | 23 |
| epiworldR | R | 1 | 62 | 66 | 4 |
| EoN | Python | 1 | 44 | 76 | 32 |
| individual | R | 1 | 37 | 80 | 43 |
| covasim | Python | 1 | 75 | 109 | 34 |
| epydemic | Python | 1 | 88 | 121 | 33 |
| starsim | Python | 1 | 94 | 136 | 42 |
| ixa | Rust | 3 | 168 | 220 | 52 |

## Interpretation

### What recording the outputs costs

For most engines, very little. The epiworld family already records all
four outputs in scenario 01, so its simulation does not change: the C++
runner takes 0.95 times as long as in scenario 01 at 100,000 agents,
epiworldR 1.00 times, and epiworldpy 1.03 times. Covasim keeps its
infection log in every run too (1.08 times). The engines that had to add
recording mostly absorbed it as well. ixa’s event subscription and
transmission log leave it at 0.70 times its scenario 01 time, and
epydemic’s handlers at 1.04 times, where the list appends are small next
to its Python event loop.

The exception is **EoN**, which takes 1.97 times as long at 10,000
agents and 1.85 times at 100,000. With `return_full_data=True`,
`fast_simple_contagion` appends to every node’s status history, in
Python, at every event.

Extraction after the run is small for every engine but epiworldR: at
100,000 agents it takes 35.0 ms there, against 8.2 ms for EoN and under
a millisecond for the others. Almost all of epiworldR’s is one call (see
below).

### Is bookkeeping what separates epiworld and ixa?

No. What epiworld records per event is small: each infection appends one
row to the transmission log, and the daily database update costs time in
proportion to the number of states and tools, not agents. ixa adds the
same bookkeeping here at little cost, and the two engines compare as
they do in scenario 01. What separates them has other causes, the same
in every scenario: ixa rebuilds its context for every replicate, while
epiworld looks up the transmission probability by name for every contact
it tries, re-initializes the whole population at the start of every run,
and, with the vaccine, clones the tool for every agent it protects.
[analysis.md](../analysis.md) measures each.

### An implementation choice that mattered: epiworldR’s reproductive-number summary

The documented way to get the daily reproductive number in epiworldR is
`plot(get_reproductive_number(model), plot = FALSE)`. The `plot()`
method computes a mean, a standard deviation, and 2.5% and 97.5%
quantiles for every day, and builds a data frame for each. That takes
about 25 ms for 100 days, no matter how small the outbreak, against 0.4
ms for a plain `tapply(rt, source_exposure_date, mean)` on the same data
frame. `get_reproductive_number()` itself takes about 1 ms. The runner
keeps the documented call, because that is what an epiworldR user would
write. A vectorized summary in epiworldR would remove nearly all of the
package’s extraction time. None of it counts toward the simulation
times.

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
