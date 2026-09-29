# Network epidemic ABM speed benchmark


- [Scenarios](#scenarios)
- [Common design](#common-design)
- [What is measured](#what-is-measured)
  - [Run time](#run-time)
  - [Measuring implementation effort](#measuring-implementation-effort)
- [Running the benchmark](#running-the-benchmark)
- [Repository layout](#repository-layout)
- [References](#references)

This project compares how fast eleven epidemic simulation engines run
the same agent-based model on the same contact network, and how much
code each engine needs to express that model. Three of them are the
epiworld family: the header-only C++ library
<a href="https://github.com/UofUEpiBio/epiworld"
target="_blank">epiworld</a> (Vega Yon 2026) and its two wrappers,
<a href="https://uofuepibio.github.io/epiworldR/"
target="_blank">epiworldR</a> (Meyer and Vega Yon 2023) for R and
<a href="https://github.com/UofUEpiBio/epiworldpy"
target="_blank">epiworldpy</a> (Vega Yon and Banks 2026) for Python. The
other eight are
<a href="https://covasim.org/" target="_blank">Covasim</a> (Kerr et al.
2021) and its successor
<a href="https://starsim.org/" target="_blank">Starsim</a> (Kerr et al.
2025),
<a href="https://epidemicsonnetworks.readthedocs.io/en/latest/EoN.html"
target="_blank">EoN</a> (Miller and Ting 2019),
<a href="https://github.com/simoninireland/epydemic"
target="_blank">epydemic</a> (Dobson 2022),
<a href="https://ixa.rs/" target="_blank">ixa</a> (The Ixa Developers
2026), the R package <a href="https://mrc-ide.github.io/individual/"
target="_blank">individual</a> (Charles and Wu 2021),
<a href="https://github.com/PublicHealthDynamicsLab/FRED"
target="_blank">FRED</a> (Grefenstette et al. 2013), and the Julia
package <a href="https://github.com/JuliaDynamics/Agents.jl"
target="_blank">Agents.jl</a> (Datseris et al. 2024).

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
| 02 | Scenario 01 plus four outputs: transmission tree, daily incidence, reproductive number, and transition matrix | [scenario_02/README.md](scenario_02/README.md) |
| 03 | Scenario 00 at 1,000,000 agents | [scenario_03/README.md](scenario_03/README.md) |
| 04 | Scenario 00 on a real contact network collapsed from <a href="https://github.com/GeoPopsHub" target="_blank">GeoPops</a> | [scenario_04/README.md](scenario_04/README.md) |

[analysis.md](analysis.md) explains why ixa is faster than the epiworld
family, with measurements that apply to every scenario.

**MEmilio was evaluated and excluded for now.** Its ABM has no edge-list
network primitive comparable to the other engines’ graphs — agents
interact only through shared Locations (household, work, school, …), so
matching scenario 04’s network would need on the order of 376,000
synthetic two-person Locations, of unproven performance, and its disease
model is an 8-compartment, viral-load-driven simulation with no simple
mapping onto the common SEIRH model used everywhere else in this
benchmark. Comparing it directly right now would need substantially more
runner code than any other engine here and would be hard to calibrate
fairly against the rest. See [issue
\#11](https://github.com/UofUEpiBio/epiworld-benchmark/issues/11) for
the design problem and a possible follow-up.

[scenarios.md](scenarios.md) explains how to add a scenario.

## Common design

Every scenario shares the following design, set in `config.toml`.

- **Population and network.** 10,000 and 100,000 agents, or 1,000,000 in
  scenario 03, on a Watts–Strogatz contact network with mean degree 10
  and rewiring probability 0.05. At each size, every engine reads the
  exact same cached edge list and treats it as an undirected graph. The
  network fixes mean degree rather than literal graph density, which
  keeps the edge count, run time, and memory from growing quadratically
  with population size.
- **Replicates.** 100 independent replicates of 100 days per engine,
  size, and scenario, and 20 in scenario 03, where the slowest engines
  take tens of seconds per replicate. Replicate seeds do not depend on
  the scenario, so two scenarios can be compared replicate by replicate.
- **Transmission.** Each engine targets an early-outbreak $R_0 = 2$
  through a shared analytic mapping from $R_0$, mean degree, and
  infectious period to a per-contact transmission rate. The engines have
  different transmission semantics (synchronous daily steps, continuous
  time, or Covasim’s native model), so a scenario may set empirical,
  size-specific transmission multipliers that align the engines’
  realized attack rates. Each scenario’s report states and justifies its
  factors.
- **Engines.** The epiworld family, epydemic, ixa, individual, and
  Agents.jl use synchronous daily transitions. FRED also steps daily,
  but an agent that becomes infectious transmits the same day. The three
  epiworld runners build the same model on the same C++ core; in
  scenario 00 they produce identical epidemics for each seed, so their
  run-time differences come from the language layer, not the model. EoN
  uses continuous-time hazards simulated exactly with its event-driven
  `fast_simple_contagion` algorithm. Covasim runs its native model,
  restricted as far as its public API allows. Starsim runs its native
  SEIR model with a hospital state added, stepping daily with sampled
  (exponential) durations. The reports therefore measure representative
  framework throughput under aligned network and disease targets, not
  bit-for-bit epidemiological equivalence.

## What is measured

### Run time

Each replicate runs in a fresh process and records:

- `simulate_seconds`, the primary measure: the wall-clock time of
  everything an engine has to redo to produce another replicate on the
  same network. That is the engine’s simulation call, including any
  initialization it does inside that call, and anything else that cannot
  be reused between runs:
  - setting up each run’s initial conditions, which differ from run to
    run (seeding the initial infections and, where there is one,
    distributing a vaccine), since epiworld does it inside `run()`;
  - for ixa, building the context (population, network, and index),
    because `execute()` runs a context once;
  - for Starsim, building and initializing the `Sim`, which can also run
    only once;
  - for FRED, which runs one replicate per process, building its places,
    population, and network from its input files (parsing those files
    counts as reading, like the other runners’ edge-list parsing).

  epiworld’s `run()` re-initializes its population every time for the
  same reason, so every engine pays for starting a fresh run. Any
  bookkeeping an engine does while it simulates is included; reading
  results out of the engine afterwards is not (scenario 02 records that
  separately as `extract_seconds`);
- `read_seconds`: reading and parsing the shared edge list, which
  depends on each language’s file handling rather than on the engine;
- `setup_seconds`: reading the edge list plus building what the engine
  can reuse across replicates, such as epiworld’s model or EoN’s graph.
  The reports show the building part (`setup_seconds - read_seconds`) as
  *build*, and build + simulate as the time to a first result once the
  edge list is in memory;
- `total_seconds`: setup plus simulation inside the runner process.

Interpreter and package startup and network generation are excluded.
Published results are collected one replicate at a time, with native
math libraries pinned to one thread, inside the container defined in
`.devcontainer/`. Running replicates concurrently, even two at a time,
slowed some engines by up to 30% and others not at all; see
[setup.md](setup.md#resource-policy).

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
    analysis.md          why ixa is faster than epiworld, across scenarios
    analysis/            the side measurements behind analysis.md

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
Flexible and General Agent-Based Model Engine*. V. 0.17.0. Released.
<https://github.com/UofUEpiBio/epiworld>.

</div>

<div id="ref-vegayonEpiworldpy2026" class="csl-entry">

Vega Yon, George G., and Olivia Banks. 2026.
*<span class="nocase">epiworldpy</span>: Python Bindings for Epiworld*.
V. 0.17.0-0. Released. <https://github.com/UofUEpiBio/epiworldpy>.

</div>

</div>
