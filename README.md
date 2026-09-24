# Network epidemic ABM speed benchmark


- [Scenarios](#scenarios)
- [Common design](#common-design)
- [What is measured](#what-is-measured)
  - [Run time](#run-time)
  - [Measuring implementation effort](#measuring-implementation-effort)
- [Running the benchmark](#running-the-benchmark)
- [Repository layout](#repository-layout)
- [References](#references)

This project compares how fast five epidemic simulation engines run the
same agent-based model on the same contact network, and how much code
each engine needs to express that model. The engines are
<a href="https://uofuepibio.github.io/epiworldR/"
target="_blank">epiworldR</a> (Meyer and Vega Yon 2023),
<a href="https://covasim.org/" target="_blank">Covasim</a> (Kerr et al.
2021),
<a href="https://epidemicsonnetworks.readthedocs.io/en/latest/EoN.html"
target="_blank">EoN</a> (Miller and Ting 2019),
<a href="https://github.com/simoninireland/epydemic"
target="_blank">epydemic</a> (Dobson 2022), and
<a href="https://ixa.rs/" target="_blank">ixa</a> (The Ixa Developers
2026).

The benchmark is organized as **scenarios** of increasing complexity.
Each scenario adds a feature to an earlier one, so together they show
how run time and implementation effort grow as the model grows. Each
scenario has its own folder, and its own report with the specification,
results, and interpretation.

## Scenarios

| Scenario | Model | Report |
|:---|:---|:---|
| 00 | SEIRH epidemic with a hospitalization branch, no interventions | [scenario_00/README.md](scenario_00/README.md) |
| 01 | Scenario 00 plus a day-0 all-or-nothing vaccine (30% coverage, 80% efficacy) | [scenario_01/README.md](scenario_01/README.md) |

[scenarios.md](scenarios.md) explains how to add a scenario.

## Common design

Every scenario shares the following design, set in `config.toml`.

- **Population and network.** 10,000 and 100,000 agents on a
  Watts–Strogatz contact network with mean degree 10 and rewiring
  probability 0.05. At each size, every engine reads the exact same
  cached edge list and treats it as an undirected graph. The network
  fixes mean degree rather than literal graph density, which keeps the
  edge count, run time, and memory from growing quadratically with
  population size.
- **Replicates.** 100 independent replicates of 100 days per engine,
  size, and scenario. Replicate seeds do not depend on the scenario, so
  two scenarios can be compared replicate by replicate.
- **Transmission.** Each engine targets an early-outbreak $R_0 = 2$
  through a shared analytic mapping from $R_0$, mean degree, and
  infectious period to a per-contact transmission rate. The engines have
  different transmission semantics (synchronous daily steps, continuous
  time, or Covasim’s native model), so a scenario may set empirical,
  size-specific transmission multipliers that align the engines’
  realized attack rates. Each scenario’s report states and justifies its
  factors.
- **Engines.** epiworldR, epydemic, and ixa use synchronous daily
  transitions. EoN uses continuous-time hazards simulated exactly with
  its event-driven `fast_simple_contagion` algorithm. Covasim runs its
  native model, restricted as far as its public API allows. The reports
  therefore measure representative framework throughput under aligned
  network and disease targets, not bit-for-bit epidemiological
  equivalence.

## What is measured

### Run time

Each replicate runs in a fresh process and records:

- `simulate_seconds`, the primary measure: wall-clock time inside the
  engine’s simulation call, including engine-native initialization that
  happens inside that call;
- `setup_seconds`: reading the shared edge list and building engine
  objects;
- `total_seconds`: setup plus simulation inside the runner process.

Interpreter and package startup and network generation are excluded.
Published results are collected one replicate at a time, with native
math libraries pinned to one thread, inside the container defined in
`.devcontainer/`. Concurrent replicates distort timings unevenly across
engines; see [setup.md](setup.md#resource-policy).

### Measuring implementation effort

Each report also counts how much code each engine needs to express the
scenario. *Model lines* counts non-blank, non-comment lines that build
the population and network, define the disease and any interventions,
and run the simulation. Argument parsing, edge-file reading, timing, and
writing results are excluded because every runner shares them. *Files*
counts the files a user writes, including build manifests. The counted
regions of each runner are listed in the scenario’s `code_regions.yml`,
and the counts are recomputed from the runner sources whenever a report
is rendered.

Each engine implements a scenario with its own native tools, such as
built-in interventions, compartments, or agent properties, so the counts
reflect what a user of that engine would write.

## Running the benchmark

The benchmark runs inside a container that pins every engine’s
toolchain. The only host requirement is podman (or Docker, with
`CONTAINER=docker`):

``` sh
make container-image
make container-check
make container-smoke
make container-benchmark     # every scenario; SCENARIOS=scenario_01 for a subset
make container-report        # renders every scenario's README.md
```

Results are cached one replicate at a time, so an interrupted run
resumes where it stopped. [setup.md](setup.md) covers the container,
running natively, targeted runs, the cache, and the resource policy.

## Repository layout

    config.toml          shared study, network, resource, and smoke settings
    run.py, scripts/     orchestration, network generation, result collection
    report/common.R      tables and figures shared by the scenario reports
    scenario_NN/
      README.qmd         the scenario's report (rendered to README.md)
      scenario.toml      model parameters and calibration
      code_regions.yml   regions counted as model code
      runners/           one runner per engine
    results/             collected results for every scenario

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
