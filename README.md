# Network epidemic ABM speed benchmark


- [Scenarios](#scenarios)
- [Summary](#summary)
- [Running the benchmark](#running-the-benchmark)
- [Repository layout](#repository-layout)
- [References](#references)

This project compares how fast thirteen epidemic simulation engines run
the same agent-based model on the same contact network, and how much
code each engine needs to express that model. Three of them are the
epiworld family: the header-only C++ library
<a href="https://github.com/UofUEpiBio/epiworld"
target="_blank">epiworld</a> (Vega Yon 2026) and its two wrappers,
<a href="https://uofuepibio.github.io/epiworldR/"
target="_blank">epiworldR</a> (Meyer and Vega Yon 2023) for R and
<a href="https://github.com/UofUEpiBio/epiworldpy"
target="_blank">epiworldpy</a> (Vega Yon and Banks 2026) for Python. The
other ten are <a href="https://covasim.org/" target="_blank">Covasim</a>
(Kerr et al. 2021) and its successor
<a href="https://starsim.org/" target="_blank">Starsim</a> (Kerr et al.
2025),
<a href="https://epidemicsonnetworks.readthedocs.io/en/latest/EoN.html"
target="_blank">EoN</a> (Miller and Ting 2019),
<a href="https://github.com/simoninireland/epydemic"
target="_blank">epydemic</a> (Dobson 2022),
<a href="https://ixa.rs/" target="_blank">ixa</a> (The Ixa Developers
2026), the R package <a href="https://mrc-ide.github.io/individual/"
target="_blank">individual</a> (Charles and Wu 2021), the R package
<a href="https://cran.r-project.org/package=ABM" target="_blank">ABM</a>
(Ma 2025), the R package
<a href="https://www.epimodel.org/" target="_blank">EpiModel</a>
(Jenness et al. 2018),
<a href="https://github.com/PublicHealthDynamicsLab/FRED"
target="_blank">FRED</a> (Grefenstette et al. 2013), and the Julia
package <a href="https://github.com/JuliaDynamics/Agents.jl"
target="_blank">Agents.jl</a> (Datseris et al. 2024).

The benchmark is organized as **scenarios** of increasing complexity.
Each scenario adds a feature to an earlier one, so together they show
how run time, memory, and implementation size grow as the model grows.
Every engine runs the same SEIRH model on the same contact network
within each scenario, one replicate at a time, inside one container
image.

- [**Methods**](docs/methods.md): the engines, the common model,
  networks, calibration, the execution protocol, and what is measured.
- [**Engines**](docs/engines.md): what each of the engines is, what it
  was built for, and how it represents a model.
- [**Engines considered and not included**](docs/excluded-engines.md):
  the inclusion criteria and why each other engine was left out.
- [**Results**](docs/results.md): run time, scaling, the cost of each
  added feature, memory, epidemiological agreement, and implementation
  size, across every scenario.
- [**Showcase models**](docs/showcase-models.md): the plan for adding,
  for each engine, a scenario it was built for.
- [**Running the benchmark**](setup.md): the container, commands, and
  the published records. [scenarios.md](scenarios.md) explains how to
  add a scenario.

## Scenarios

| Scenario | Model | Report |
|:---|:---|:---|
| 00 | SEIRH epidemic with a hospitalization branch, no interventions | [scenario_00/README.md](scenario_00/README.md) |
| 01 | Scenario 00 plus a day-0 all-or-nothing vaccine (30% coverage, 80% efficacy) | [scenario_01/README.md](scenario_01/README.md) |
| 02 | Scenario 01 plus four outputs: transmission tree, daily incidence, reproductive number, and transition matrix | [scenario_02/README.md](scenario_02/README.md) |
| 03 | Scenario 00 at 1,000,000 agents | [scenario_03/README.md](scenario_03/README.md) |
| 04 | Scenario 00 on a real contact network collapsed from <a href="https://github.com/GeoPopsHub" target="_blank">GeoPops</a> | [scenario_04/README.md](scenario_04/README.md) |

## Summary

Medians across replicates, with each scenario at its largest population
size (in parentheses). Scenarios 00 to 02 also run at 10,000 agents; the
[results](docs/results.md) and each scenario’s report show every size,
interquartile ranges, and the other phases of a run. Engines are ordered
from fastest (or smallest) overall at the top; the axes are logarithmic,
because the engines differ by orders of magnitude.

**Simulation time**: everything an engine redoes for each replicate on
the same network ([Timing](docs/methods.md#timing)).

![Median simulation time per replicate (seconds, log scale), by engine
and scenario.](README_files/figure-commonmark/summary-time-plot-1.png)

<details>

<summary>

The same medians as a table (seconds)
</summary>

| Engine     | 00 (100k) | 01 (100k) | 02 (100k) | 03 (1M) | 04 (165,865) |
|:-----------|----------:|----------:|----------:|--------:|-------------:|
| epiworldpy |   0.00732 |   0.00597 |   0.00602 |  0.0271 |        0.152 |
| epiworldR  |     0.008 |     0.007 |     0.007 |  0.0235 |        0.157 |
| epiworld   |   0.00772 |    0.0072 |   0.00714 |  0.0257 |        0.156 |
| ixa        |    0.0334 |    0.0289 |    0.0284 |   0.373 |        0.149 |
| Agents.jl  |    0.0366 |    0.0374 |    0.0366 |   0.727 |        0.133 |
| individual |     0.053 |     0.049 |      0.07 |   0.272 |        0.136 |
| covasim    |     0.324 |     0.334 |     0.334 |    3.14 |        0.562 |
| EoN\*      |     0.285 |     0.211 |      0.45 |    2.27 |         3.53 |
| FRED       |     0.485 |     0.587 |     0.706 |     7.2 |         2.66 |
| starsim    |     0.905 |     0.908 |     0.715 |    11.0 |        0.822 |
| ABM\*      |     0.936 |     0.868 |     0.827 |    11.0 |         14.3 |
| epydemic   |      1.71 |      1.07 |      1.08 |    13.3 |         12.5 |
| EpiModel   |      7.03 |      6.96 |      7.66 |    79.3 |         5.91 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

</details>

**Memory**: the overall peak resident memory of a replicate’s process
([Memory](docs/methods.md#memory)).

![Median overall peak memory per replicate (MiB, log scale), by engine
and scenario.](README_files/figure-commonmark/summary-memory-plot-1.png)

<details>

<summary>

The same medians as a table (MiB)
</summary>

| Engine     | 00 (100k) | 01 (100k) | 02 (100k) | 03 (1M) | 04 (165,865) |
|:-----------|----------:|----------:|----------:|--------:|-------------:|
| ixa        |        42 |        42 |        42 |     409 |           46 |
| epiworld   |        38 |        43 |        43 |     352 |           90 |
| individual |       127 |       127 |       142 |     282 |          130 |
| epiworldR  |       113 |       113 |       113 |     479 |          150 |
| epiworldpy |       127 |       132 |       132 |     854 |          137 |
| ABM\*      |       192 |       191 |       191 |   1,344 |          313 |
| covasim    |       272 |       281 |       281 |     684 |          307 |
| starsim    |       339 |       339 |       341 |     905 |          331 |
| EoN\*      |       284 |       286 |       313 |   1,671 |          350 |
| epydemic   |       413 |       413 |       413 |   3,020 |          416 |
| EpiModel   |       485 |       487 |       477 |   1,893 |          495 |
| Agents.jl  |       574 |       585 |       586 |     971 |          569 |
| FRED       |       581 |       597 |       599 |   5,847 |          784 |

<sub>\* Continuous-time engine: it simulates events at exact times, and
the benchmark records the daily totals.</sub>

</details>

## Running the benchmark

The only host requirement is podman (or Docker, with
`CONTAINER=docker`):

``` sh
make container-image
make container-check
make container-smoke
make container-benchmark     # every scenario; SCENARIOS=scenario_01 for a subset
make container-report        # this overview, docs/, and every scenario report
```

## Repository layout

    config.toml          shared study, network, resource, and smoke settings
    run.py, scripts/     orchestration, network generation, environment records
    report/common.R      tables and figures shared by every report
    docs/
      methods.qmd        how the benchmark is built and run (rendered to .md)
      results.qmd        results across scenarios (rendered to .md)
      epiworld-ixa.md    where epiworld's and ixa's time goes
      engines.md         what each engine is (rendered from engines.qmd)
      excluded-engines.md engines left out and why (rendered from excluded-engines.qmd)
      showcase-models.md the plan for a scenario per engine
    scenario_NN/
      README.qmd         the scenario's report (rendered to README.md)
      scenario.toml      model parameters and calibration
      code_regions.yml   regions counted as model code
      runners/           one runner per engine
    results/
      runs/              one record per replicate (the source of every table)
      environments/      where each scenario's latest run ran
      results.csv, daily.csv, scenarios.json   derived from runs/
    analysis/            the side measurements behind docs/epiworld-ixa.md

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-charlesIndividual2021" class="csl-entry">

Charles, Giovanni D., and Sean L. Wu. 2021. “Individual: An R Package
for Individual-Based Epidemiological Models.” *Journal of Open Source
Software* 6 (66): 3539. <https://doi.org/10.21105/joss.03539>.

</div>

<div id="ref-datserisAgentsjl2024" class="csl-entry">

Datseris, George, Ali R. Vahdati, and Timothy C. DuBois. 2024.
“Agents.jl: A Performant and Feature-Full Agent-Based Modeling Software
of Minimal Code Complexity.” *SIMULATION* 100 (10): 1019–31.
<https://doi.org/10.1177/00375497211068820>.

</div>

<div id="ref-dobson2022epydemic" class="csl-entry">

Dobson, Simon. 2022. “Epydemic: Epidemic Simulation on Networks in
Python.” In *GitHub Repository*.
<a href="https://github.com/simoninireland/epydemic"
class="uri">Https://github.com/simoninireland/epydemic</a>; GitHub.

</div>

<div id="ref-grefenstetteFRED2013" class="csl-entry">

Grefenstette, John J., Shawn T. Brown, Roni Rosenfeld, et al. 2013.
“FRED (<span class="nocase">A Framework for Reconstructing Epidemic
Dynamics</span>): An Open-Source Software System for Modeling Infectious
Diseases and Control Strategies Using Census-Based Populations.” *BMC
Public Health* 13: 940. <https://doi.org/10.1186/1471-2458-13-940>.

</div>

<div id="ref-jennessEpiModel2018" class="csl-entry">

Jenness, Samuel M., Steven M. Goodreau, and Martina Morris. 2018.
“EpiModel: An R Package for Mathematical Modeling of Infectious Disease
over Networks.” *Journal of Statistical Software* 84 (8).
<https://doi.org/10.18637/jss.v084.i08>.

</div>

<div id="ref-kerrCovasimAgentbasedModel2021" class="csl-entry">

Kerr, Cliff C., Robyn M. Stuart, Dina Mistry, et al. 2021. “Covasim: An
Agent-Based Model of COVID-19 Dynamics and Interventions.” *PLOS
Computational Biology* 17 (7): e1009149.
<https://doi.org/10.1371/journal.pcbi.1009149>.

</div>

<div id="ref-kerrStarsim2025" class="csl-entry">

Kerr, Cliff, Robyn Stuart, Romesh Abeysuriya, et al. 2025. *Starsim*. V.
3.0.0. Released July. <https://github.com/starsimhub/starsim>.

</div>

<div id="ref-maABM2025" class="csl-entry">

Ma, Junling. 2025. *ABM: Agent Based Model Simulation Framework*.
<https://CRAN.R-project.org/package=ABM>.

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

<div id="ref-vegayonEpiworld2026" class="csl-entry">

Vega Yon, George G. 2026. *<span class="nocase">epiworld</span>: A
Flexible and General Agent-Based Model Engine*. V. 0.17.1. Released.
<https://github.com/UofUEpiBio/epiworld>.

</div>

<div id="ref-vegayonEpiworldpy2026" class="csl-entry">

Vega Yon, George G., and Olivia Banks. 2026.
*<span class="nocase">epiworldpy</span>: Python Bindings for Epiworld*.
V. 0.17.1-0. Released. <https://github.com/UofUEpiBio/epiworldpy>.

</div>

</div>
