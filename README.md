# Network epidemic ABM speed benchmark

2026-09-23

- [Executive summary](#executive-summary)
- [Simulation time](#simulation-time)
- [Speed relative to epiworldR](#speed-relative-to-epiworldr)
- [Epidemiological sanity checks](#epidemiological-sanity-checks)
- [Design and interpretation](#design-and-interpretation)
- [Reproducibility and cache](#reproducibility-and-cache)
- [References](#references)

## Executive summary

This benchmark compares simulation speed for 100-day network epidemics
at 10,000 and 100,000 agents, with 100 independent replicates per engine
and size. The included engines are
<a href="https://uofuepibio.github.io/epiworldR/"
target="_blank">epiworldR</a> (Meyer and Vega Yon 2023),
<a href="https://covasim.org/" target="_blank">Covasim</a> (Kerr et al.
2021),
<a href="https://epidemicsonnetworks.readthedocs.io/en/latest/EoN.html"
target="_blank">EoN</a> (Miller and Ting 2019),
<a href="https://github.com/simoninireland/epydemic"
target="_blank">epydemic</a> (Dobson 2022), and
<a href="https://ixa.rs/" target="_blank">ixa</a> (The Ixa Developers
2026). All engines use the exact same cached Watts–Strogatz edge list
for a population size (mean degree 10, rewiring probability 0.05) and
target an early-outbreak $R_0 = 2$. The epiworldR cells apply
size-specific transmission multipliers of 0.887 at 10,000 agents and
0.855 at 100,000 agents. The restricted Covasim cells apply
corresponding factors of 0.715 and 0.710, and the ixa cells apply 0.983
and 0.988. These empirical factors align realized attack rates under the
frameworks’ different transmission semantics.

> [!TIP]
>
> The complete 1,000-run design is available.

| Field               | Value                                                 |
|:--------------------|:------------------------------------------------------|
| Platform            | Linux-6.12.13-200.fc41.aarch64-aarch64-with-glibc2.39 |
| Python              | 3.12.11                                               |
| Workers             | 6                                                     |
| Latest run failures | 0                                                     |

| Engine    | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:----------|-------:|-----:|----------------------:|-------:|-------:|
| ixa       |  10000 |  100 |                 0.006 |  0.005 |  0.007 |
| epiworldR |  10000 |  100 |                 0.041 |  0.031 |  0.054 |
| covasim   |  10000 |  100 |                 0.119 |  0.097 |  0.156 |
| EoN       |  10000 |  100 |                 0.124 |  0.110 |  0.164 |
| epydemic  |  10000 |  100 |                 0.732 |  0.657 |  0.836 |
| ixa       | 100000 |  100 |                 0.012 |  0.010 |  0.015 |
| epiworldR | 100000 |  100 |                 0.170 |  0.129 |  0.242 |
| EoN       | 100000 |  100 |                 0.552 |  0.490 |  0.615 |
| covasim   | 100000 |  100 |                 0.643 |  0.511 |  0.860 |
| epydemic  | 100000 |  100 |                 3.275 |  2.848 |  3.581 |

## Simulation time

The primary measure is wall-clock time inside each engine’s simulation
call. The logarithmic scale keeps fast and slow engines legible in one
panel.

![](README_files/figure-commonmark/simulation-time-plot-1.png)

## Speed relative to epiworldR

Ratios are matched by population size and replicate seed. Values above
one mean that epiworldR completed the simulation call faster.

| Engine   | Agents | Median time / epiworldR |    Q1 |    Q3 |
|:---------|-------:|------------------------:|------:|------:|
| covasim  |  10000 |                    2.67 |  2.03 |  4.20 |
| EoN      |  10000 |                    3.16 |  2.22 |  4.41 |
| epydemic |  10000 |                   17.57 | 12.85 | 25.00 |
| ixa      |  10000 |                    0.16 |  0.12 |  0.20 |
| covasim  | 100000 |                    3.99 |  2.43 |  5.40 |
| EoN      | 100000 |                    3.43 |  2.46 |  4.61 |
| epydemic | 100000 |                   19.45 | 12.15 | 28.65 |
| ixa      | 100000 |                    0.07 |  0.04 |  0.11 |

## Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show final attack rate and peak hospitalization
load; differences also expose the engines’ non-identical time semantics
and Covasim’s native symptom/severe bookkeeping.

| Engine    | Agents | Median final attack rate | Median peak hospitalized |
|:----------|-------:|-------------------------:|-------------------------:|
| epiworldR |  10000 |                    0.380 |                     22.0 |
| covasim   |  10000 |                    0.392 |                     23.0 |
| EoN       |  10000 |                    0.379 |                     21.0 |
| epydemic  |  10000 |                    0.396 |                     25.0 |
| ixa       |  10000 |                    0.378 |                     18.5 |
| epiworldR | 100000 |                    0.062 |                     28.0 |
| covasim   | 100000 |                    0.061 |                     35.0 |
| EoN       | 100000 |                    0.060 |                     32.0 |
| epydemic  | 100000 |                    0.064 |                     41.0 |
| ixa       | 100000 |                    0.062 |                     31.0 |

![](README_files/figure-commonmark/outcomes-plot-1.png)

## Design and interpretation

The common disease graph is susceptible $\rightarrow$ exposed
$\rightarrow$ infectious $\rightarrow$ recovered, with a competing
infectious $\rightarrow$ hospitalized $\rightarrow$ recovered branch.
Mean latent and infectious periods are 4 and 7 days, lifetime
hospitalization probability is 5%, and the hospital stay is 7 days.
Initial infections are 100 in the full profile.

The shared analytic transmission mapping produced different realized
attack rates across frameworks. Calibration therefore selected
size-specific factors for epiworldR (0.887 at 10,000 agents and 0.855 at
100,000), for restricted Covasim (0.715 and 0.710), and for ixa (0.983
and 0.988, matching its median attack rate to the epiworldR cells). All
100 replicates in each adjusted cell were then rerun. These empirical
corrections are intentionally population-specific; they align benchmark
outcomes but mean that none of the calibrated engines has a strictly
analytic $R_0$ of 2. The values live in `config.toml` and participate in
the cache fingerprint.

epiworldR, epydemic, and ixa use synchronous daily transitions. The ixa
runner schedules one plan per day on ixa’s plan queue, samples
transmission along ixa’s built-in contact-network edges, and stores
disease status as an indexed entity property; its competing infectious
$\rightarrow$ hospitalized/recovered step reuses epiworldR’s roulette
rule. EoN uses continuous-time hazards, simulated exactly with its
event-driven `fast_simple_contagion` algorithm. Covasim retains its
native exposed, infectious, symptomatic, and severe bookkeeping, with
severe prevalence used as the hospitalization proxy. Its runner disables
waning immunity and fixes individual transmissibility and viral load to
one, making recovered people permanently removed and removing those
sources of heterogeneity. Covasim does not expose a public switch to
remove the remaining symptom/severity bookkeeping. Thus this report
measures restricted, representative framework throughput under aligned
network and disease targets; it does not claim bit-for-bit
epidemiological equivalence.

The sparse contact network fixes mean degree rather than literal graph
density. At 10,000 and 100,000 agents its densities are approximately
0.001 and 0.0001, respectively, while every agent still has about ten
contacts. This prevents the edge count, runtime, and memory from growing
quadratically.

More details regarding the project setup can be found in
[setup.md](./setup.md).

## Reproducibility and cache

Every successful replicate is an atomic JSON cache record keyed by model
configuration, runner source, engine version, and contact-network
SHA-256. Rerunning `make benchmark` schedules only missing or stale
records. The default worker count is one, and common native math-library
thread counts are pinned to one. Concurrency is opt-in through the
`N_THREADS` environment variable. Concurrent replicates contend for
memory bandwidth and performance cores, and the penalty differs by
engine, so timings are comparable only across runs at the same
concurrency. The platform and worker count for the run behind this
report are shown in the table above; the benchmark runs inside the
container defined in `.devcontainer/` (see [setup.md](./setup.md)),
which pins every engine’s toolchain.

| Engine    | Recorded version |
|:----------|:-----------------|
| covasim   | 3.1.8            |
| EoN       | 1.92             |
| epiworldR | 0.15.1.0         |
| epydemic  | 1.14.1           |
| ixa       | 3.1.0            |

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-dobson2022epydemic" class="csl-entry">

Dobson, Simon. 2022. “Epydemic: Epidemic Simulation on Networks in
Python.” In *GitHub Repository*.
<a href="https://github.com/simoninireland/epydemic"
class="uri">Https://github.com/simoninireland/epydemic</a>; GitHub.

</div>

<div id="ref-kerrCovasimAgentbasedModel2021" class="csl-entry">

Kerr, Cliff C., Robyn M. Stuart, Dina Mistry, et al. 2021. “Covasim: An
Agent-Based Model of COVID-19 Dynamics and Interventions.” *PLOS
Computational Biology* 17 (7): e1009149.
<https://doi.org/10.1371/journal.pcbi.1009149>.

</div>

<div id="ref-meyerEpiworldRFastAgentBased2023" class="csl-entry">

Meyer, Derek, and George G Vega Yon. 2023.
“<span class="nocase">epiworldR</span>: Fast Agent-Based Epi Models.”
*Journal of Open Source Software* 8 (90): 5781.
<https://doi.org/10.21105/joss.05781>.

</div>

<div id="ref-millerEoNEpidemicsNetworks2019" class="csl-entry">

Miller, Joel, and Tony Ting. 2019. “EoN (Epidemics on Networks): A Fast,
Flexible Python Package for Simulation, Analytic Approximation, and
Analysis of Epidemics on Networks.” *Journal of Open Source Software* 4
(44): 1731. <https://doi.org/10.21105/joss.01731>.

</div>

<div id="ref-ixa2026" class="csl-entry">

The Ixa Developers. 2026. *<span class="nocase">ixa</span>: A Framework
for Building Agent-Based Models*. V. 3.1.0. Centers for Disease Control;
Prevention, Center for Forecasting; Outbreak Analytics, released.
<https://github.com/CDCgov/ixa>.

</div>

</div>
