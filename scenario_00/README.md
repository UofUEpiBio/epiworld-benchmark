# Scenario 00: SEIRH baseline

2026-09-29

- [Model](#model)
  - [Engine implementations](#engine-implementations)
- [Results](#results)
  - [Simulation time](#simulation-time)
  - [Speed relative to epiworldR](#speed-relative-to-epiworldr)
  - [Time to a first result](#time-to-a-first-result)
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
family, Starsim, and FRED. epiworldR’s API cannot seed a custom model’s
initial cases in any state other than the one new infections enter, so
they start exposed there. The C++ and Python runners do the same, so
that all three epiworld runners build exactly the same model. Starsim’s
SEIR puts its seed cases where new infections go, which is also exposed,
and so does FRED’s import.

The simulation time covers everything an engine has to redo for another
replicate on the same network (see [what is
measured](../README.md#run-time)). Seeding the initial cases is part of
it in every runner, because the seeds differ in every run and epiworld
places them inside `run()`, which also re-initializes the whole
population. EoN’s timer covers building its initial status dictionary,
individual’s covers building its state variable, and the other engines
seed inside their simulation call. ixa’s `execute()` runs a context only
once, so its timer also covers building the context: the population, the
network, and the index on disease status; it then seeds in a plan at
time 0. Starsim’s `Sim` also runs only once, so its timer covers
building the disease, network, and `Sim` objects and `sim.init()`, which
builds the agent and network arrays and seeds the initial cases. FRED
runs one replicate per process and rebuilds everything each time; its
timer covers building its places, population, and network as well as the
days (see its engine description below).

[Scenario 03](../scenario_03/README.md) runs this model at 1,000,000
agents.

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
- **Starsim**: its built-in `ss.SEIR` disease, subclassed to add a
  hospital state, on a network module holding the shared edge list.
  Transmission is Starsim’s own per-edge, per-day draw with a daily
  probability (`ss.probperday`). Durations are sampled when an agent
  enters a state: exponential latent, infectious, and hospital periods,
  with hospitalization decided by a Bernoulli draw with the lifetime
  probability.
- **EoN**: a continuous-time rate graph run with its event-driven
  `fast_simple_contagion` algorithm.
- **epydemic**: a `CompartmentedModel` under `SynchronousDynamics`.
- **ixa**: one plan per day, ixa’s built-in contact network, and an
  indexed disease-status property. The competing infectious transitions
  reuse epiworldR’s roulette rule.
- **individual**: a `CategoricalVariable` for the disease state, the
  built-in `bernoulli_process()` for E $\rightarrow$ I and H
  $\rightarrow$ R, and `fixed_probability_multinomial_process()` for
  leaving I, to H with the lifetime hospitalization probability and
  otherwise to R. individual has no contact networks, so infection is a
  process written for this benchmark over an adjacency list built from
  the edge list: a susceptible agent with $k$ infectious neighbours is
  infected with probability $1 - (1 - \beta)^k$, as in epiworld. Updates
  queued during a day apply at its end, so the steps are synchronous.
- **FRED** (<a href="https://github.com/PublicHealthDynamicsLab/FRED"
  target="_blank">PublicHealthDynamicsLab/FRED</a>): a compiled
  simulator driven by its own model language, with no edge-list network
  primitive comparable to the other engines’ graphs. The runner
  ([`run_FRED.py`](runners/run_FRED.py)) writes a synthetic population
  of single-person households (so FRED’s own household mixing adds no
  contacts beyond the network) and loads the benchmark’s edges verbatim
  as `Contact.add_edge` properties on FRED’s `Network` group type, with
  transmission restricted to that network. The disease is a FRED
  condition with one state per compartment, and FRED’s `import_count()`
  exposes the seed cases on day 0. Each day, an infectious agent draws
  about `transmissibility` × degree contacts without replacement among
  its neighbours, so every neighbour is reached with that daily
  probability, and the runner uses the same per-day mapping as the
  synchronous engines. Latent, infectious, and hospital stays are whole
  days with the configured means. FRED’s `days` counts day 0, which here
  only places the seed cases, so the runner asks for one more day than
  the other engines run. The build patches an upstream off-by-one in
  `Date::setup_dates()` that corrupts the heap at large populations (see
  the [Dockerfile](../.devcontainer/Dockerfile)). FRED runs one
  replicate per process and must reread and rebuild everything each
  time. Its timings are split with FRED’s own lap timers: parsing the
  generated model file (which holds every edge) and the population files
  counts as reading, like the other runners’ edge-file parsing; building
  places, the population, and the network, plus the days themselves,
  count as simulation, as ixa’s and Starsim’s per-replicate builds do.
- **Agents.jl** (<a href="https://github.com/JuliaDynamics/Agents.jl"
  target="_blank">JuliaDynamics/Agents.jl</a>): a `StandardABM` with a
  per-agent status and a per-day `model_step!` that scans each
  infectious agent’s adjacency list, matching the synchronous daily
  semantics of epiworldR, epydemic, and individual. The adjacency list
  and transmission math are written for this benchmark, since Agents.jl
  has no built-in epidemic model or contact network. Before any timer
  starts, the runner steps a throwaway two-agent model so that Julia
  compiles its methods; otherwise about 0.1 seconds of compilation lands
  in the first simulation.

## Results

> [!TIP]
>
> The complete 2,200-run design for this scenario is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 1                                                     |
| Latest run failures | 0                                                     |

| Engine     | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:-----------|-------:|-----:|----------------------:|-------:|-------:|
| ixa        |  10000 |  100 |                 0.007 |  0.007 |  0.008 |
| epiworldpy |  10000 |  100 |                 0.008 |  0.008 |  0.009 |
| epiworld   |  10000 |  100 |                 0.009 |  0.008 |  0.010 |
| epiworldR  |  10000 |  100 |                 0.009 |  0.008 |  0.009 |
| Agents.jl  |  10000 |  100 |                 0.021 |  0.020 |  0.024 |
| individual |  10000 |  100 |                 0.026 |  0.025 |  0.027 |
| covasim    |  10000 |  100 |                 0.065 |  0.064 |  0.067 |
| FRED       |  10000 |  100 |                 0.072 |  0.069 |  0.077 |
| EoN        |  10000 |  100 |                 0.079 |  0.074 |  0.085 |
| starsim    |  10000 |  100 |                 0.174 |  0.169 |  0.182 |
| epydemic   |  10000 |  100 |                 0.509 |  0.483 |  0.552 |
| epiworldpy | 100000 |  100 |                 0.014 |  0.013 |  0.016 |
| epiworldR  | 100000 |  100 |                 0.015 |  0.013 |  0.016 |
| epiworld   | 100000 |  100 |                 0.015 |  0.013 |  0.016 |
| ixa        | 100000 |  100 |                 0.038 |  0.035 |  0.045 |
| Agents.jl  | 100000 |  100 |                 0.042 |  0.039 |  0.046 |
| individual | 100000 |  100 |                 0.060 |  0.056 |  0.070 |
| EoN        | 100000 |  100 |                 0.306 |  0.293 |  0.329 |
| covasim    | 100000 |  100 |                 0.329 |  0.324 |  0.348 |
| FRED       | 100000 |  100 |                 0.544 |  0.524 |  0.589 |
| starsim    | 100000 |  100 |                 0.938 |  0.907 |  0.998 |
| epydemic   | 100000 |  100 |                 1.854 |  1.747 |  1.974 |

### Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

### Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine     | Agents | Median time / epiworldR |     Q1 |     Q3 |
|:-----------|-------:|------------------------:|-------:|-------:|
| epiworld   |  10000 |                    1.00 |   0.97 |   1.06 |
| epiworldpy |  10000 |                    0.94 |   0.91 |   0.98 |
| covasim    |  10000 |                    7.35 |   7.03 |   8.10 |
| starsim    |  10000 |                   19.52 |  18.46 |  22.14 |
| EoN        |  10000 |                    8.93 |   8.12 |  10.15 |
| epydemic   |  10000 |                   59.48 |  53.26 |  64.10 |
| ixa        |  10000 |                    0.81 |   0.76 |   0.91 |
| individual |  10000 |                    3.00 |   2.78 |   3.22 |
| FRED       |  10000 |                    8.16 |   7.43 |   8.98 |
| Agents.jl  |  10000 |                    2.50 |   2.25 |   2.86 |
| epiworld   | 100000 |                    1.02 |   0.99 |   1.06 |
| epiworldpy | 100000 |                    0.96 |   0.91 |   0.99 |
| covasim    | 100000 |                   22.89 |  20.91 |  25.00 |
| starsim    | 100000 |                   65.11 |  59.51 |  73.53 |
| EoN        | 100000 |                   21.36 |  18.93 |  24.83 |
| epydemic   | 100000 |                  124.36 | 115.90 | 144.54 |
| ixa        | 100000 |                    2.64 |   2.39 |   3.00 |
| individual | 100000 |                    4.19 |   3.67 |   5.23 |
| FRED       | 100000 |                   38.32 |  33.83 |  42.39 |
| Agents.jl  | 100000 |                    2.92 |   2.58 |   3.29 |

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
| ixa        |  10000 |          0.003 |     0.000 |        0.007 |                0.007 |
| epiworld   |  10000 |          0.005 |     0.001 |        0.009 |                0.010 |
| epiworldpy |  10000 |          0.016 |     0.004 |        0.008 |                0.012 |
| epiworldR  |  10000 |          0.008 |     0.015 |        0.009 |                0.024 |
| Agents.jl  |  10000 |          0.156 |     0.019 |        0.021 |                0.041 |
| individual |  10000 |          0.008 |     0.015 |        0.026 |                0.041 |
| covasim    |  10000 |          0.017 |     0.041 |        0.065 |                0.106 |
| EoN        |  10000 |          0.017 |     0.032 |        0.079 |                0.110 |
| starsim    |  10000 |          0.017 |     0.000 |        0.174 |                0.174 |
| FRED       |  10000 |          0.260 |     0.119 |        0.072 |                0.193 |
| epydemic   |  10000 |          0.017 |     0.018 |        0.509 |                0.529 |
| epiworld   | 100000 |          0.049 |     0.012 |        0.015 |                0.027 |
| ixa        | 100000 |          0.029 |     0.000 |        0.038 |                0.039 |
| epiworldR  | 100000 |          0.069 |     0.026 |        0.015 |                0.041 |
| epiworldpy | 100000 |          0.153 |     0.036 |        0.014 |                0.050 |
| Agents.jl  | 100000 |          0.318 |     0.025 |        0.042 |                0.067 |
| individual | 100000 |          0.072 |     0.031 |        0.060 |                0.090 |
| covasim    | 100000 |          0.153 |     0.041 |        0.329 |                0.372 |
| EoN        | 100000 |          0.161 |     0.274 |        0.306 |                0.587 |
| starsim    | 100000 |          0.152 |     0.000 |        0.938 |                0.938 |
| FRED       | 100000 |          2.618 |     0.868 |        0.544 |                1.423 |
| epydemic   | 100000 |          0.156 |     0.312 |        1.854 |                2.160 |

### The epiworld family

The three epiworld runners build the same model on the same C++ core,
and in this scenario they produce identical epidemics for every seed.
This table compares the two wrappers with the C++ runner, replicate by
replicate.

| Engine     | Agents | Median time / epiworld |   Q1 |   Q3 |
|:-----------|-------:|-----------------------:|-----:|-----:|
| epiworldR  |  10000 |                   1.00 | 0.94 | 1.03 |
| epiworldpy |  10000 |                   0.93 | 0.91 | 0.97 |
| epiworldR  | 100000 |                   0.98 | 0.95 | 1.01 |
| epiworldpy | 100000 |                   0.94 | 0.88 | 0.98 |

### Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show the final attack rate and peak
hospitalization load. Differences also expose the engines’ non-identical
time semantics and Covasim’s native symptom/severe bookkeeping.

| Engine     | Agents | Median final attack rate | Median peak hospitalized |
|:-----------|-------:|-------------------------:|-------------------------:|
| Agents.jl  |  10000 |                    0.387 |                     18.0 |
| covasim    |  10000 |                    0.392 |                     23.0 |
| EoN        |  10000 |                    0.379 |                     21.0 |
| epiworld   |  10000 |                    0.385 |                     19.0 |
| epiworldpy |  10000 |                    0.385 |                     19.0 |
| epiworldR  |  10000 |                    0.385 |                     19.0 |
| epydemic   |  10000 |                    0.396 |                     25.0 |
| FRED       |  10000 |                    0.385 |                     21.0 |
| individual |  10000 |                    0.381 |                     20.5 |
| ixa        |  10000 |                    0.378 |                     18.5 |
| starsim    |  10000 |                    0.387 |                     21.0 |
| Agents.jl  | 100000 |                    0.062 |                     30.0 |
| covasim    | 100000 |                    0.061 |                     35.0 |
| EoN        | 100000 |                    0.060 |                     32.0 |
| epiworld   | 100000 |                    0.060 |                     30.0 |
| epiworldpy | 100000 |                    0.060 |                     30.0 |
| epiworldR  | 100000 |                    0.060 |                     30.0 |
| epydemic   | 100000 |                    0.064 |                     41.0 |
| FRED       | 100000 |                    0.060 |                     33.0 |
| individual | 100000 |                    0.060 |                     34.0 |
| ixa        | 100000 |                    0.062 |                     31.0 |
| starsim    | 100000 |                    0.059 |                     33.5 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

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
| covasim    | Python   |     1 |          64 |
| Agents.jl  | Julia    |     1 |          66 |
| epydemic   | Python   |     1 |          70 |
| starsim    | Python   |     1 |          83 |
| FRED       | Python   |     1 |          92 |
| ixa        | Rust     |     3 |         138 |

Model lines vary with how much of the model an engine provides built in.
For example, EoN describes transitions as rate graphs, Covasim needs its
native parameters overridden to restrict it to SEIRH, Starsim needs a
subclass of its SEIR to add the hospital state, and the ixa runner
writes its daily step by hand, as the individual and Agents.jl runners
write their infection processes. The FRED runner writes FRED’s model
file and synthetic population from Python, so its lines are the Python
that generates them. ixa needs a Cargo project that is compiled before
it runs, and FRED a compiled binary.

## Interpretation

### Calibration

The shared analytic transmission mapping produced different realized
attack rates across frameworks. Calibration therefore selected
size-specific factors for restricted Covasim (0.715 at 10,000 agents and
0.710 at 100,000), ixa (0.983 and 0.988), individual (0.951 at both),
FRED (0.911 and 0.898), and Agents.jl (0.981 and 0.982). epiworldR needs
none: its uncalibrated median attack rates match EoN’s exact
continuous-time results, and so do epiworld’s and epiworldpy’s, which
run the same model. Starsim needs none either (uncalibrated medians
0.387 and 0.060). All 100 replicates in each adjusted cell were then
rerun. These empirical corrections are intentionally
population-specific; they align benchmark outcomes but mean that none of
the calibrated engines has a strictly analytic $R_0$ of 2. The values
live in [`scenario.toml`](scenario.toml) and participate in the cache
fingerprint.

Covasim’s factors were calibrated after restricting it to fixed
transmission and permanent immunity. ixa shares the synchronous daily
semantics of epiworldR and epydemic, but its uncalibrated attack rates
sat above the other engines: it seeds the initial cases as infectious
and samples each infectious contact independently rather than with
epiworld’s roulette. individual’s sat higher still (0.441 and 0.074). It
too seeds the initial cases as infectious, and its infectious period
averages the full seven days, where the epiworld family’s competing
daily hospitalization rate shortens it slightly. Agents.jl seeds the
initial cases as infectious too, and its uncalibrated medians (0.414 and
0.065) sat slightly above the target, like ixa’s. FRED’s sat far above
(0.519 and 0.095). Its contact draw reaches each neighbour with the same
daily probability as the synchronous engines, but an agent that becomes
infectious at the start of a day transmits that same day, where a
synchronous engine’s new case first transmits on the next step. Each
generation is therefore about a day shorter, and within 100 days the
outbreak grows further. FRED and Agents.jl were calibrated against
epiworldR’s medians over 100 replicates per candidate factor, outside
the timed runs.

### The language layer costs nothing measurable in the simulation

The three epiworld runners build the same model and run the same
epidemics, and they take the same time within noise at both sizes (see
the table above). Earlier versions of this report found the C++ runner
slower than the wrappers at 100,000 agents. That came from how epiworld
built the network, not from the language layer, and it went away with
the counting-sort network build
([UofUEpiBio/epiworld#274](https://github.com/UofUEpiBio/epiworld/issues/274)).

Outside the simulation, the wrappers cost something. At 100,000 agents,
building the model from the edge list takes the C++ runner 0.012
seconds, against 0.026 for epiworldR and 0.036 for epiworldpy, which
first convert the edge list into their own types. Reading the edge file
also differs by language, but that is the benchmark’s plumbing rather
than the engine’s, so the time-to-first-result table leaves it out.

### Why the two fastest engines differ

epiworld 0.16.1 made network transmission push-based when that is
cheaper: infectious agents push infection to their neighbours instead of
every queued susceptible agent scanning all of its neighbours for
infectious ones (the change proposed in
[UofUEpiBio/epiworld#264](https://github.com/UofUEpiBio/epiworld/issues/264)).
Together with the other changes through 0.17.0, this made epiworldR
about four times faster here than 0.15.1.0 was at 100,000 agents. Both
engines’ daily work now follows the outbreak rather than the population.

Per replicate, ixa is a little faster at 10,000 agents, and epiworld is
faster at 100,000: ixa takes 0.8 and 2.5 times as long as the C++
runner. ixa’s `execute()` can run a context only once, so every
replicate rebuilds the population, network, and index, which grows with
the population. epiworld builds its model once and resets it at the
start of every `run()`, which is about twenty times cheaper. ixa’s
`execute()` alone is faster than epiworld’s `run()`, because epiworld
looks up the transmission probability by name for every contact it tries
and re-initializes every agent in `reset()`.
[analysis.md](../analysis.md) measures each of these, and shows how they
map onto every scenario.

### Equivalence across engines

epiworldR, epydemic, ixa, individual, and Agents.jl use synchronous
daily transitions, while EoN uses continuous-time hazards, simulated
exactly. FRED steps daily too, but applies each day’s transitions before
that day’s transmission, as described above. Starsim also steps daily,
but samples each agent’s time in a state when it enters it, and applies
transitions before transmission within a step. Covasim retains its
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

| Engine     | Recorded version   |
|:-----------|:-------------------|
| Agents.jl  | 7.0.4              |
| covasim    | 3.1.8              |
| EoN        | 1.92               |
| epiworld   | 0.17.0             |
| epiworldpy | 0.17.0-0+g4a1ee0b  |
| epiworldR  | 0.17.0.0           |
| epydemic   | 1.14.1             |
| FRED       | PUB.5.7.0+gbd25f04 |
| individual | 0.1.19             |
| ixa        | 3.1.0              |
| starsim    | 3.6.1              |
