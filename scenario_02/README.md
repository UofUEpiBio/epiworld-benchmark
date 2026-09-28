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
  - [Is bookkeeping why epiworld is slower than
    ixa?](#is-bookkeeping-why-epiworld-is-slower-than-ixa)
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

The EoN, epydemic, Covasim, and ixa runners use epiworld’s definition of
the reproductive number. Each Python runner has its own copy of a small
helper that computes it from the tree, and that helper counts as model
code.

## Results

> [!TIP]
>
> The complete 1,400-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 4                                                     |
| Latest run failures | 0                                                     |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| ixa        |  10000 |  100 |                 0.002 |  0.002 |  0.003 |
| epiworldpy |  10000 |  100 |                 0.005 |  0.004 |  0.006 |
| epiworldR  |  10000 |  100 |                 0.005 |  0.005 |  0.006 |
| epiworld   |  10000 |  100 |                 0.005 |  0.005 |  0.007 |
| covasim    |  10000 |  100 |                 0.076 |  0.073 |  0.089 |
| EoN        |  10000 |  100 |                 0.078 |  0.075 |  0.082 |
| epydemic   |  10000 |  100 |                 0.256 |  0.241 |  0.278 |
| ixa        | 100000 |  100 |                 0.005 |  0.005 |  0.006 |
| epiworldR  | 100000 |  100 |                 0.011 |  0.010 |  0.012 |
| epiworldpy | 100000 |  100 |                 0.013 |  0.011 |  0.016 |
| epiworld   | 100000 |  100 |                 0.014 |  0.013 |  0.017 |
| covasim    | 100000 |  100 |                 0.421 |  0.398 |  0.442 |
| EoN        | 100000 |  100 |                 0.905 |  0.707 |  1.099 |
| epydemic   | 100000 |  100 |                 1.687 |  1.440 |  2.047 |

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
| epiworld   |  10000 |                    0.99 |   0.82 |   1.38 |
| epiworldpy |  10000 |                    0.91 |   0.74 |   1.13 |
| covasim    |  10000 |                   14.68 |  12.24 |  18.06 |
| EoN        |  10000 |                   14.88 |  12.57 |  16.35 |
| epydemic   |  10000 |                   49.29 |  38.98 |  57.86 |
| ixa        |  10000 |                    0.42 |   0.36 |   0.49 |
| epiworld   | 100000 |                    1.30 |   1.14 |   1.57 |
| epiworldpy | 100000 |                    1.14 |   0.95 |   1.39 |
| covasim    | 100000 |                   38.68 |  33.37 |  43.42 |
| EoN        | 100000 |                   77.40 |  64.95 |  95.93 |
| epydemic   | 100000 |                  155.38 | 123.23 | 217.15 |
| ixa        | 100000 |                    0.46 |   0.39 |   0.55 |

### Extracting the outputs

The time taken to read the four outputs out of each engine after the
run. It is not part of the simulation times above.

| Engine | Agents | Median simulation (s) | Median extraction (s) | Median extraction / simulation |
|:---|---:|---:|---:|---:|
| ixa | 10000 | 0.002 | 0.000 | 0.04 |
| epiworldpy | 10000 | 0.005 | 0.001 | 0.11 |
| epiworldR | 10000 | 0.005 | 0.035 | 6.30 |
| epiworld | 10000 | 0.005 | 0.000 | 0.04 |
| covasim | 10000 | 0.076 | 0.000 | 0.00 |
| EoN | 10000 | 0.078 | 0.001 | 0.02 |
| epydemic | 10000 | 0.256 | 0.000 | 0.00 |
| ixa | 100000 | 0.005 | 0.000 | 0.02 |
| epiworldR | 100000 | 0.011 | 0.031 | 2.80 |
| epiworldpy | 100000 | 0.013 | 0.001 | 0.05 |
| epiworld | 100000 | 0.014 | 0.000 | 0.02 |
| covasim | 100000 | 0.421 | 0.001 | 0.00 |
| EoN | 100000 | 0.905 | 0.011 | 0.01 |
| epydemic | 100000 | 1.687 | 0.001 | 0.00 |

### Epidemiological sanity checks

| Engine | Agents | Median final attack rate | Median peak hospitalized | Median share protected |
|:---|---:|---:|---:|---:|
| covasim | 10000 | 0.105 | 11 | 0.24 |
| EoN | 10000 | 0.113 | 10 | 0.24 |
| epiworld | 10000 | 0.119 | 8 | 0.24 |
| epiworldpy | 10000 | 0.117 | 8 | 0.24 |
| epiworldR | 10000 | 0.115 | 9 | 0.24 |
| epydemic | 10000 | 0.121 | 11 | 0.24 |
| ixa | 10000 | 0.119 | 8 | 0.24 |
| covasim | 100000 | 0.012 | 11 | 0.24 |
| EoN | 100000 | 0.013 | 10 | 0.24 |
| epiworld | 100000 | 0.014 | 9 | 0.24 |
| epiworldpy | 100000 | 0.014 | 9 | 0.24 |
| epiworldR | 100000 | 0.014 | 9 | 0.24 |
| epydemic | 100000 | 0.014 | 11 | 0.24 |
| ixa | 100000 | 0.014 | 9 | 0.24 |

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
| EoN | 10000 | 1034 | 1 | 1 |
| epydemic | 10000 | 1108 | 1 | 1 |
| ixa | 10000 | 1088 | 1 | 1 |
| epiworld | 100000 | 1288 | 1 | 1 |
| epiworldR | 100000 | 1293 | 1 | 1 |
| epiworldpy | 100000 | 1264 | 1 | 1 |
| covasim | 100000 | 1084 | 1 | 1 |
| EoN | 100000 | 1188 | 1 | 1 |
| epydemic | 100000 | 1281 | 1 | 1 |
| ixa | 100000 | 1269 | 1 | 1 |

The daily series, as medians across replicates. The epiworld family
seeds its initial cases as exposed, and so does Covasim, so their
incidence starts a few days later than that of the engines that seed
them as infected.

![](README_files/figure-commonmark/daily-plot-1.png)

## Cost of the outputs

### Run time compared with scenario 01

Each replicate is paired with the scenario 01 replicate that has the
same engine, population size, and seed. Both scenarios run the same
epidemic, so a ratio above one is the cost of recording the outputs
during the run.

| Engine | Agents | Median scenario 01 (s) | Median scenario 02 (s) | Median time / scenario 01 | Q1 | Q3 |
|:---|---:|---:|---:|---:|---:|---:|
| epiworldpy | 10000 | 0.005 | 0.005 | 0.97 | 0.84 | 1.08 |
| epydemic | 10000 | 0.259 | 0.256 | 0.98 | 0.90 | 1.06 |
| epiworldR | 10000 | 0.005 | 0.005 | 1.00 | 0.82 | 1.25 |
| ixa | 10000 | 0.002 | 0.002 | 1.00 | 0.91 | 1.10 |
| epiworld | 10000 | 0.005 | 0.005 | 1.07 | 0.99 | 1.32 |
| covasim | 10000 | 0.070 | 0.076 | 1.13 | 1.01 | 1.28 |
| EoN | 10000 | 0.039 | 0.078 | 1.99 | 1.87 | 2.10 |
| epiworldR | 100000 | 0.015 | 0.011 | 0.77 | 0.55 | 0.92 |
| covasim | 100000 | 0.420 | 0.421 | 0.98 | 0.92 | 1.07 |
| epiworld | 100000 | 0.014 | 0.014 | 1.03 | 0.94 | 1.19 |
| epiworldpy | 100000 | 0.012 | 0.013 | 1.04 | 0.93 | 1.20 |
| ixa | 100000 | 0.005 | 0.005 | 1.07 | 0.94 | 1.19 |
| epydemic | 100000 | 1.518 | 1.687 | 1.12 | 0.99 | 1.26 |
| EoN | 100000 | 0.273 | 0.905 | 3.14 | 2.62 | 3.83 |

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
| covasim | Python | 1 | 75 | 109 | 34 |
| epydemic | Python | 1 | 88 | 121 | 33 |
| ixa | Rust | 3 | 168 | 220 | 52 |

## Interpretation

### What recording the outputs costs

For most engines, very little. The epiworld family already records all
four outputs in scenario 01, so its simulation does not change: the C++
runner takes 1.03 times as long as in scenario 01 at 100,000 agents,
epiworldR 0.77 times, and epiworldpy 1.04 times. Covasim keeps its
infection log in every run too (0.98 times). The engines that had to add
recording mostly absorbed it as well. ixa’s event subscription and
transmission log leave it at 1.07 times its scenario 01 time, and
epydemic’s handlers at 1.12 times, where the list appends are small next
to its Python event loop.

The exception is **EoN**, which takes 1.99 times as long at 10,000
agents and 3.14 times at 100,000. With `return_full_data=True`,
`fast_simple_contagion` appends to every node’s status history, in
Python, at every event.

Extraction after the run is small for every engine but epiworldR: at
100,000 agents it takes 31.0 ms there, against 10.9 ms for EoN and under
a millisecond for the others. Almost all of epiworldR’s is one call (see
below).

### Is bookkeeping why epiworld is slower than ixa?

No. What epiworld records per event is small: each infection appends one
row to the transmission log, and the daily database update costs time in
proportion to the number of states and tools, not agents. ixa does the
same bookkeeping here at almost no cost. epiworld’s queue also works as
intended: with no infected agents, its 100 days take about 0.2 ms at
100,000 agents.

Side measurements with the C++ runner, outside the benchmark, at 100,000
agents, found where its time goes instead:

- **About 11 ms is a memory-allocator effect.** `agents_from_edgelist()`
  builds a temporary adjacency list with one `std::map` per agent,
  roughly a million small allocations. When it is freed, glibc keeps
  those chunks on its fast free lists, and the next large allocation
  walks them all to consolidate them (`malloc_consolidate`). That
  allocation happens inside `run()`, when the state index is built, so
  the timed call pays for tearing down the network builder. Disabling
  glibc’s fast bins (`GLIBC_TUNABLES=glibc.malloc.mxfast=0`) cuts the
  C++ runner from about 21 ms to about 9.5 ms, in line with epiworldR,
  whose setup triggers the consolidation before the timer starts.
  Reported upstream as
  [UofUEpiBio/epiworld#274](https://github.com/UofUEpiBio/epiworld/issues/274),
  with a fix that builds the network about ten times faster and removes
  the charge.
- **About 2.7 ms is placing the vaccine** on about 24,000 agents. Every
  runner now times its vaccine distribution, so this is a like-for-like
  cost.
- **About 1.5 ms is looking up the transmission probability.** Push
  transmission costs about 45 ns per neighbour it checks. For every one,
  the virus reads `"Transmission rate"` through `Model::get_param()`,
  which takes the name by value (17 characters, so a heap allocation)
  and searches a `std::map` twice, although the value depends only on
  the infector. With a constant probability the same check costs about
  23 ns.
- **The rest, about 2 ms,** is the other work in `reset()`, the E, I,
  and H transitions, and the vaccine check on each neighbour, which goes
  through the agent’s tools and their `std::function`s.

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
| ixa        | 3.1.0             |
