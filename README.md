# Network epidemic ABM speed benchmark

2026-09-13

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
target="_blank">EoN</a> (Miller and Ting 2019), and
<a href="https://github.com/simoninireland/epydemic"
target="_blank">epydemic</a> (Dobson 2022). All engines use the exact
same cached Watts–Strogatz edge list for a population size (mean degree
10, rewiring probability 0.05) and target an early-outbreak $R_0 = 2$.
The epiworldR cells apply size-specific transmission multipliers of
0.887 at 10,000 agents and 0.855 at 100,000 agents. The restricted
Covasim cells apply corresponding factors of 0.715 and 0.710. These
empirical factors align realized attack rates under the frameworks’
different transmission semantics.

> [!TIP]
>
> The complete 800-run design is available.

| Field               | Value                        |
|:--------------------|:-----------------------------|
| Platform            | macOS-15.0.1-arm64-arm-64bit |
| Python              | 3.12.11                      |
| Workers             | 4                            |
| Latest run failures | 0                            |

| Engine    | Agents | Runs | Median simulation (s) | Q1 (s) | Q3 (s) |
|:----------|-------:|-----:|----------------------:|-------:|-------:|
| epiworldR |  10000 |  100 |                 0.038 |  0.036 |  0.040 |
| covasim   |  10000 |  100 |                 0.101 |  0.099 |  0.104 |
| EoN       |  10000 |  100 |                 0.112 |  0.106 |  0.120 |
| epydemic  |  10000 |  100 |                 0.694 |  0.659 |  0.719 |
| epiworldR | 100000 |  100 |                 0.083 |  0.076 |  0.096 |
| EoN       | 100000 |  100 |                 0.437 |  0.417 |  0.450 |
| covasim   | 100000 |  100 |                 0.558 |  0.542 |  0.598 |
| epydemic  | 100000 |  100 |                 2.387 |  2.295 |  2.502 |

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
| covasim  |  10000 |                    2.72 |  2.53 |  2.87 |
| EoN      |  10000 |                    2.93 |  2.69 |  3.17 |
| epydemic |  10000 |                   18.31 | 16.34 | 19.82 |
| covasim  | 100000 |                    6.81 |  5.91 |  7.43 |
| EoN      | 100000 |                    5.22 |  4.47 |  5.68 |
| epydemic | 100000 |                   28.89 | 24.89 | 31.90 |

## Epidemiological sanity checks

Speed is interpretable only if the simulations produce plausible
epidemics. These checks show final attack rate and peak hospitalization
load; differences also expose the engines’ non-identical time semantics
and Covasim’s native symptom/severe bookkeeping.

| Engine    | Agents | Median final attack rate | Median peak hospitalized |
|:----------|-------:|-------------------------:|-------------------------:|
| epiworldR |  10000 |                    0.380 |                       22 |
| covasim   |  10000 |                    0.392 |                       23 |
| EoN       |  10000 |                    0.379 |                       21 |
| epydemic  |  10000 |                    0.396 |                       25 |
| epiworldR | 100000 |                    0.062 |                       28 |
| covasim   | 100000 |                    0.061 |                       35 |
| EoN       | 100000 |                    0.060 |                       32 |
| epydemic  | 100000 |                    0.064 |                       41 |

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
100,000) and for restricted Covasim (0.715 and 0.710). All 100
replicates in each adjusted cell were then rerun. These empirical
corrections are intentionally population-specific; they align benchmark
outcomes but mean that neither calibrated engine has a strictly analytic
$R_0$ of 2. The values live in `config.toml` and participate in the
cache fingerprint.

epiworldR and epydemic use synchronous daily transitions. EoN uses
continuous-time hazards, simulated exactly with its event-driven
`fast_simple_contagion` algorithm. Covasim retains its native exposed,
infectious, symptomatic, and severe bookkeeping, with severe prevalence
used as the hospitalization proxy. Its runner disables waning immunity
and fixes individual transmissibility and viral load to one, making
recovered people permanently removed and removing those sources of
heterogeneity. Covasim does not expose a public switch to remove the
remaining symptom/severity bookkeeping. Thus this report measures
restricted, representative framework throughput under aligned network
and disease targets; it does not claim bit-for-bit epidemiological
equivalence.

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
`N_THREADS` environment variable and is intended for filling the cache,
not for measurement: concurrent replicates contend for memory bandwidth
and performance cores, and the penalty differs by engine, so the timings
reported here are collected sequentially. The worker count for the run
behind this report is shown in the table above.

| Engine    | Recorded version |
|:----------|:-----------------|
| covasim   | 3.1.8            |
| EoN       | 1.92             |
| epiworldR | 0.15.1.0         |
| epydemic  | 1.14.1           |

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

</div>
