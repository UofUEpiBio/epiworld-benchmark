# Scenario 00: SEIRH baseline

2026-09-26

- [Model](#model)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [The epiworld family](#the-epiworld-family)
  - [Epidemiological sanity checks](#epidemiological-sanity-checks)
  - [Code required to define the
    model](#code-required-to-define-the-model)
- [Interpretation](#interpretation)
  - [Calibration](#calibration)
  - [The language layer costs nothing measurable in the
    simulation](#the-language-layer-costs-nothing-measurable-in-the-simulation)
  - [Why the two fastest engines
    differ](#why-the-two-fastest-engines-differ)
  - [Equivalence across engines](#equivalence-across-engines)
- [Recorded versions](#recorded-versions)

[Back to the project overview](../README.md)

The reference model for the benchmark. Every later scenario adds
complexity to it, and each later scenario’s report compares its run time
and model code against this one.

## Model

Susceptible $\rightarrow$ exposed $\rightarrow$ infectious $\rightarrow$
recovered, with a competing infectious $\rightarrow$ hospitalized
$\rightarrow$ recovered branch. Transmission happens along the shared
Watts–Strogatz contact network (mean degree 10, rewiring probability
0.05). There are no interventions.

| Parameter (`scenario.toml`) | Value | Meaning |
|:---|---:|:---|
| `target_r0` | 2.0 | Early-outbreak $R_0$ used by the analytic mapping |
| `initial_infected` | 100 | Seed cases (10 in the smoke profile) |
| `latent_days` | 4.0 | Mean exposed period |
| `infectious_days` | 7.0 | Mean infectious period |
| `hospitalization_probability` | 0.05 | Lifetime probability that an infectious agent is hospitalized |
| `hospital_days` | 7.0 | Mean hospital stay |

Initial cases start infectious in every engine except the epiworld
family. epiworldR’s API cannot seed a custom model’s initial cases in
any state other than the one new infections enter, so they start exposed
there. The C++ and Python runners do the same, so that all three
epiworld runners build exactly the same model.

### Engine implementations

- **epiworld** (C++): a custom model built from `add_state()` and the
  library’s update-function factories,
  `sampler::make_update_susceptible()` and
  `new_state_update_transition()`. New infections enter Exposed. It is
  compiled with the flags R uses for epiworldR (`-O2 -DNDEBUG`, double
  precision).
- **epiworldR**: the same model through the R wrapper, with
  `update_fun_susceptible()` and `update_fun_rate()`.
- **epiworldpy**: the same model through the Python wrapper, with
  `UpdateFun.susceptible()` and `UpdateFun.rate()`. These were added for
  this benchmark in
  [UofUEpiBio/epiworldpy#17](https://github.com/UofUEpiBio/epiworldpy/pull/17).
  Before, a custom model’s other states needed Python callbacks, which
  epiworld calls for every agent every day. epiworldpy compiles epiworld
  with its default single-precision `float`, and with the same seed it
  still gives the same epidemics as the other two.
- **Covasim**: native disease progression restricted to SEIRH. Waning is
  off, and individual transmissibility and viral load are fixed. The
  severe state serves as the hospitalization proxy.
- **EoN**: a continuous-time rate graph run with its event-driven
  `fast_simple_contagion` algorithm.
- **epydemic**: a `CompartmentedModel` under `SynchronousDynamics`.
- **ixa**: one plan per day, ixa’s built-in contact network, and an
  indexed disease-status property. The competing infectious transitions
  reuse epiworldR’s roulette rule.

## Results

> [!TIP]
>
> The complete 1,400-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 1                                                     |
| Latest run failures | 0                                                     |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| ixa        |  10000 |  100 |                 0.004 |  0.004 |  0.005 |
| epiworldpy |  10000 |  100 |                 0.008 |  0.008 |  0.009 |
| epiworldR  |  10000 |  100 |                 0.009 |  0.008 |  0.009 |
| epiworld   |  10000 |  100 |                 0.009 |  0.009 |  0.010 |
| covasim    |  10000 |  100 |                 0.064 |  0.063 |  0.066 |
| EoN        |  10000 |  100 |                 0.077 |  0.072 |  0.080 |
| epydemic   |  10000 |  100 |                 0.489 |  0.468 |  0.520 |
| ixa        | 100000 |  100 |                 0.008 |  0.008 |  0.009 |
| epiworldpy | 100000 |  100 |                 0.015 |  0.013 |  0.016 |
| epiworldR  | 100000 |  100 |                 0.016 |  0.014 |  0.017 |
| epiworld   | 100000 |  100 |                 0.026 |  0.024 |  0.028 |
| EoN        | 100000 |  100 |                 0.293 |  0.280 |  0.308 |
| covasim    | 100000 |  100 |                 0.326 |  0.323 |  0.331 |
| epydemic   | 100000 |  100 |                 1.727 |  1.659 |  1.812 |

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |    Q1 |     Q3 |
|:-----------|-------:|------------------------:|------:|-------:|
| epiworld   |  10000 |                    1.08 |  1.02 |   1.14 |
| epiworldpy |  10000 |                    0.97 |  0.93 |   1.04 |
| covasim    |  10000 |                    7.61 |  7.07 |   8.12 |
| EoN        |  10000 |                    8.85 |  8.08 |   9.90 |
| epydemic   |  10000 |                   56.79 | 52.37 |  62.82 |
| ixa        |  10000 |                    0.48 |  0.44 |   0.54 |
| epiworld   | 100000 |                    1.66 |  1.58 |   1.77 |
| epiworldpy | 100000 |                    0.94 |  0.90 |   0.97 |
| covasim    | 100000 |                   21.51 | 19.10 |  23.23 |
| EoN        | 100000 |                   19.00 | 17.37 |  21.05 |
| epydemic   | 100000 |                  112.74 | 99.77 | 124.87 |
| ixa        | 100000 |                    0.53 |  0.47 |   0.59 |

### The epiworld family

The three epiworld runners build the same model on the same C++ core,
and in this scenario they produce identical epidemics for every seed.
This table compares the two wrappers with the C++ runner, replicate by
replicate.

| Engine     | Agents | Median time / epiworld |   Q1 |   Q3 |
|:-----------|-------:|-----------------------:|-----:|-----:|
| epiworldR  |  10000 |                   0.93 | 0.88 | 0.98 |
| epiworldpy |  10000 |                   0.91 | 0.89 | 0.93 |
| epiworldR  | 100000 |                   0.60 | 0.56 | 0.63 |
| epiworldpy | 100000 |                   0.57 | 0.53 | 0.59 |

### Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show the final attack rate and peak
hospitalization load. Differences also expose the engines’ non-identical
time semantics and Covasim’s native symptom/severe bookkeeping.

| Engine     | Agents | Median final attack rate | Median peak hospitalized |
|:-----------|-------:|-------------------------:|-------------------------:|
| covasim    |  10000 |                    0.392 |                     23.0 |
| EoN        |  10000 |                    0.379 |                     21.0 |
| epiworld   |  10000 |                    0.385 |                     19.0 |
| epiworldpy |  10000 |                    0.385 |                     19.0 |
| epiworldR  |  10000 |                    0.385 |                     19.0 |
| epydemic   |  10000 |                    0.396 |                     25.0 |
| ixa        |  10000 |                    0.378 |                     18.5 |
| covasim    | 100000 |                    0.061 |                     35.0 |
| EoN        | 100000 |                    0.060 |                     32.0 |
| epiworld   | 100000 |                    0.060 |                     30.0 |
| epiworldpy | 100000 |                    0.060 |                     30.0 |
| epiworldR  | 100000 |                    0.060 |                     30.0 |
| epydemic   | 100000 |                    0.064 |                     41.0 |
| ixa        | 100000 |                    0.062 |                     31.0 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

### Code required to define the model

A rough measure of effort: how much code each engine needs to express
the model. The counting rules are described in the [project
overview](../README.md#measuring-implementation-effort), and the counted
regions are listed in [`code_regions.yml`](code_regions.yml).

| Engine     | Language | Files | Model lines |
|:-----------|:---------|------:|------------:|
| epiworld   | C++      |     1 |          26 |
| epiworldpy | Python   |     1 |          34 |
| EoN        | Python   |     1 |          37 |
| epiworldR  | R        |     1 |          47 |
| covasim    | Python   |     1 |          64 |
| epydemic   | Python   |     1 |          70 |
| ixa        | Rust     |     3 |         133 |

Model lines vary with how much of the model an engine provides built in.
For example, EoN describes transitions as rate graphs, Covasim needs its
native parameters overridden to restrict it to SEIRH, and the ixa runner
writes its daily step by hand. The Python engines share one runner file,
and ixa needs a Cargo project that is compiled before it runs.

## Interpretation

### Calibration

The shared analytic transmission mapping produced different realized
attack rates across frameworks. Calibration therefore selected
size-specific factors for restricted Covasim (0.715 at 10,000 agents and
0.710 at 100,000) and for ixa (0.983 and 0.988). epiworldR needs none:
its uncalibrated median attack rates match EoN’s exact continuous-time
results, and so do epiworld’s and epiworldpy’s, which run the same
model. All 100 replicates in each adjusted cell were then rerun. These
empirical corrections are intentionally population-specific; they align
benchmark outcomes but mean that none of the calibrated engines has a
strictly analytic $R_0$ of 2. The values live in
[`scenario.toml`](scenario.toml) and participate in the cache
fingerprint.

Covasim’s factors were calibrated after restricting it to fixed
transmission and permanent immunity. ixa shares the synchronous daily
semantics of epiworldR and epydemic, but its uncalibrated attack rates
sat above the other engines: it seeds the initial cases as infectious
and samples each infectious contact independently rather than with
epiworld’s roulette.

### The language layer costs nothing measurable in the simulation

Neither wrapper is slower than the C++ runner. At 10,000 agents
epiworldR and epiworldpy are even 7-9% faster, and at 100,000 agents
they take about 0.015 seconds against the C++ runner’s 0.026, although
all three run the same compiled code on the same model. The C++ runner
is not doing more work: run twice in the same process, its second run
takes 0.016 seconds. Raising glibc’s `mmap` threshold
(`MALLOC_MMAP_THRESHOLD_`), so that large blocks come from the ordinary
heap, also brings its first run down to 0.017 seconds. The difference is
where the model’s large per-agent arrays land in memory. In the C++
runner, each array gets its own fresh, page-aligned `mmap` region. The R
and Python processes have already grown and freed a large heap while
starting up and reading the edge list, so the arrays land there instead.
The run makes no page faults in either case, which points to cache or
TLB effects of the placement rather than to first-touch cost. The runner
is timed as a C++ user would write it, so the published numbers keep
this effect. Allocating those arrays together would likely remove it
upstream.

Outside the simulation call the ordering flips. The C++ runner reads the
edge list and builds the model fastest, so its `total_seconds` (which
excludes interpreter startup) is the lowest of the three.

### Why the two fastest engines differ

epiworld 0.16.1 made network transmission push-based when that is
cheaper: infectious agents push infection to their neighbours instead of
every queued susceptible agent scanning all of its neighbours for
infectious ones (the change proposed in
[UofUEpiBio/epiworld#264](https://github.com/UofUEpiBio/epiworld/issues/264)).
Together with the other changes through 0.17.0, this made epiworldR
about four times faster here than 0.15.1.0 was at 100,000 agents (median
0.062 to 0.016 seconds). In push mode, both engines’ daily cost now
follows the outbreak rather than the population: each visits only the
exposed, infectious, and hospitalized agents, and tries transmission
outward from the infectious ones. ixa is still about twice as fast at
100,000 agents. This report does not profile where the remaining gap
comes from.

### Equivalence across engines

epiworldR, epydemic, and ixa use synchronous daily transitions, while
EoN uses continuous-time hazards, simulated exactly. Covasim retains its
native exposed, infectious, symptomatic, and severe bookkeeping, with
severe prevalence used as the hospitalization proxy. Its runner disables
waning immunity and fixes individual transmissibility and viral load to
one, making recovered people permanently removed and removing those
sources of heterogeneity. Covasim does not expose a public switch to
remove the remaining symptom/severity bookkeeping. Thus this report
measures restricted, representative framework throughput under aligned
network and disease targets; it does not claim bit-for-bit
epidemiological equivalence.

## Recorded versions

| Engine     | Recorded version  |
|:-----------|:------------------|
| covasim    | 3.1.8             |
| EoN        | 1.92              |
| epiworld   | 0.17.0            |
| epiworldpy | 0.17.0-0+g583c55e |
| epiworldR  | 0.17.0.0          |
| epydemic   | 1.14.1            |
| ixa        | 3.1.0             |
