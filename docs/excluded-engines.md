# Engines considered and not included


- [Inclusion criteria](#inclusion-criteria)
- [Status](#status)
- [Summary](#summary)
- [Excluded after assessment](#excluded-after-assessment)
  - [MEmilio](#memilio)
  - [Epiabm](#epiabm)
- [Not yet assessed](#not-yet-assessed)
- [Adding an entry](#adding-an-entry)
- [References](#references)

[Back to the project overview](../README.md) · [Methods](methods.md) ·
[Engines](engines.md) · [Showcase model plan](showcase-models.md)

This document records every simulation engine that was considered for
the benchmark but is not part of it, and why. It is written to become a
supplementary section of the manuscript: the criteria, the status of
each engine, and the reasons for exclusion are stated so that a reader
can check them and so that a later version of the benchmark can revisit
them.

## Inclusion criteria

An engine is included when it can run the benchmark’s common design
([Methods](methods.md#inclusion-and-exclusion)) without changing what
the design measures. The criteria are:

1.  **Common model.** It runs the common SEIRH model
    ([Methods](methods.md#common-model)), natively or through a
    documented mapping that leaves the compartments and their
    transitions unchanged.
2.  **Arbitrary contact network.** It takes an arbitrary undirected edge
    list as the contact structure, through a network primitive or a
    documented equivalent that does not change the number of contacts
    per agent.
3.  **Scripted and seeded.** It runs non-interactively from a script,
    with a seed the runner sets.
4.  **Compartment counts.** It reports, or lets the runner compute, the
    daily counts of the five compartments.
5.  **Open and installable.** It is open source and builds or installs
    in the benchmark’s container image on a single machine.

Criteria 1 to 4 are the ones the methods state; criterion 5 is implicit
in the execution protocol. Failing a criterion does not mean an engine
is slow or unsuitable for epidemic modeling: several of the engines
below were built for models richer than the common one, and they are
candidates for the [showcase models](showcase-models.md), which give
each engine a model it was designed for.

## Status

Each engine has one of two statuses:

- **Excluded after assessment.** The engine’s documentation and source
  were reviewed against the criteria and at least one criterion fails,
  or can only be met by a workaround whose cost would dominate what the
  benchmark measures. No runner was written; the assessment says what a
  runner would need. An exclusion holds “for now” and records what would
  reverse it.
- **Not yet assessed.** The engine was listed as a candidate ([issue
  \#7](https://github.com/UofUEpiBio/epiworld-benchmark/issues/7)), but
  no one has checked it against the criteria. Its note is an initial
  expectation, not a finding.

Candidates from that list that have since been added, FRED and
Agents.jl, are described in [Engines](engines.md).

## Summary

| Engine | Language | Status | Criteria not met | Possible route to inclusion |
|:---|:---|:---|:---|:---|
| MEmilio (Bicker et al. 2026) | C++, Python interface | Excluded after assessment | 1, 2 | Prototype one location per edge; or a showcase on its native locations |
| Epiabm (Gallagher et al. 2024) | Python and C++ | Excluded after assessment | 1, 2 | Showcase of a CovidSim-style spatial household model |
| EpiModel (Jenness et al. 2018) | R | Not yet assessed |  |  |
| OpenABM-Covid19 (Hinch et al. 2021) | C, Python interface | Not yet assessed |  |  |
| Mesa (<span class="nocase">ter Hoeven et al.</span> 2025) | Python | Not yet assessed |  |  |
| NDlib (Rossetti et al. 2018) | Python | Not yet assessed |  |  |
| LASER | Python (Numba) | Not yet assessed |  |  |
| EMOD (<span class="nocase">Bershteyn et al.</span> 2018) | C++ | Not yet assessed |  |  |
| CovidSim (<span class="nocase">Ferguson et al.</span> 2020) | C++ | Not yet assessed |  |  |
| JUNE (Aylett-Bullock et al. 2021) | Python | Not yet assessed |  |  |
| EpiHiper, Loimos, Repast HPC (Collier and North 2013) | C++, Charm++, MPI | Not yet assessed |  |  |

## Excluded after assessment

### MEmilio

MEmilio (Bicker et al. 2026) is a C++ framework with a Python interface
from the German Aerospace Center (DLR) and partners. It covers
compartmental, metapopulation, and agent-based models; the assessment
concerns its agent-based model, `mio::abm`. Assessed while adding FRED,
Agents.jl, and scenario 04 ([issue
\#11](https://github.com/UofUEpiBio/epiworld-benchmark/issues/11)).

- **Criterion 2.** The ABM has no edge-list or graph contact primitive.
  Agents interact only through shared `Location` objects (household,
  work, school, social event, …). The only faithful translation of an
  edge list is one synthetic two-person location per edge, about 376,000
  locations for scenario 04’s network. No other runner needs such a
  construction, and its performance at that scale is unknown.
- **Criterion 1.** The disease model has eight infection states (S, E,
  I_NoSymptoms, I_Symptoms, I_Severe, I_Critical, R, D) driven by
  per-person viral-load trajectories. Mapping it onto SEIRH, including
  what to do with death, which the common model lacks, and calibrating
  it to the common targets is modeling work rather than a runner port.
- **What would reverse it.** A prototype of the one-location-per-edge
  approach at a few thousand edges showing viable performance, followed
  by an explicit eight-state to SEIRH mapping. Otherwise, a showcase
  model on MEmilio’s own household, work, and school locations, stated
  as not graph-equivalent to the other engines.

### Epiabm

Epiabm (Gallagher et al. 2024) re-implements the Imperial College
CovidSim model (<span class="nocase">Ferguson et al.</span> 2020) with
an emphasis on readable code. It has a Python backend (pyEpiabm) for
small populations and teaching, and a C++ backend (cEpiabm) for
populations the size of a country. Assessed from its documentation and
source at release v1.2.1 (June 2025).

- **Criterion 2.** Contacts come from a fixed spatial hierarchy of
  cells, microcells, households, and places, plus a distance kernel
  between cells, each implemented as a “sweep” (household, place,
  spatial). There is no edge-list input. pyEpiabm accepts user-written
  sweeps, so a network sweep could be written, but pyEpiabm is
  documented as unable to handle large populations, and cEpiabm would
  need the sweep in C++.
- **Criterion 1.** Infection states are a fixed set from CovidSim:
  susceptible, exposed, asymptomatic, mild, GP, hospitalized, ICU, ICU
  recovery, recovered, dead, and vaccinated. SEIRH would be a collapse
  of this progression rather than a configuration of it.
- **Criterion 5.** Met, with caveats: pyEpiabm is not on PyPI and
  installs from source; cEpiabm builds with CMake.
- **What would reverse it.** Its strength is the CovidSim structure:
  households, places, age-dependent mixing, spatial spread, and
  interventions. That makes it a candidate for a showcase model next to
  FRED’s and Covasim’s ([showcase models](showcase-models.md)) rather
  than for the common scenarios.

## Not yet assessed

The notes come from the candidate list in [issue
\#7](https://github.com/UofUEpiBio/epiworld-benchmark/issues/7) and
state what an assessment should check first. They are expectations, not
findings.

- **EpiModel** (Jenness et al. 2018). The most used network epidemic
  package in R and the R peer to epiworldR. Check whether a fixed edge
  list can be used in place of the temporal exponential random graph
  models it is built around, and the cost of a custom SEIRH module.
- **OpenABM-Covid19** (Hinch et al. 2021). A fast network ABM with
  built-in multi-layer networks, testing, and tracing. Check whether a
  user-supplied edge list can replace its generated networks.
- **Mesa** (<span class="nocase">ter Hoeven et al.</span> 2025). The
  most used general ABM framework in Python, expected to be slow but
  representative of what many users try first. A NetworkX graph should
  satisfy criterion 2.
- **NDlib** (Rossetti et al. 2018). Diffusion models on NetworkX graphs.
  Check whether SEIRH needs a custom model.
- **LASER** (<https://github.com/laser-base/laser-core>). The Institute
  for Disease Modeling’s framework for very large populations. Young,
  with an evolving API.
- **EMOD** (<span class="nocase">Bershteyn et al.</span> 2018). The
  Institute for Disease Modeling’s production model;
  configuration-driven. Check its contact representation.
- **CovidSim** (<span class="nocase">Ferguson et al.</span> 2020).
  Spatial and household-based, like Epiabm, which re-implements it;
  expected to fail criterion 2 for the same reasons. An assessment
  should decide whether Epiabm stands in for it.
- **JUNE** (Aylett-Bullock et al. 2021). A detailed model tied to UK
  geography.
- **EpiHiper, Loimos, Repast HPC** (Collier and North 2013). Engines for
  populations of billions of agents on clusters. Expected to fail
  criterion 5 as stated (one machine), and to answer a different
  question than this benchmark.

## Adding an entry

When an engine is considered, add a row to the summary table and, once
it has been assessed, a subsection under “Excluded after assessment”
with: the engine’s version or commit and what was read (documentation,
source, a prototype); each criterion that fails and why, with the
evidence; and what would reverse the decision. Link the issue where the
assessment was discussed, and add the engine’s reference to
`references.bib`. An engine that is added to the benchmark moves to
[Engines](engines.md).

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-aylettbullockJUNE2021" class="csl-entry">

Aylett-Bullock, Joseph, Carolina Cuesta-Lazaro, Arnau Quera-Bofarull, et
al. 2021. “JUNE: Open-Source Individual-Based Epidemiology Simulation.”
*Royal Society Open Science* 8 (7): 210506.
<https://doi.org/10.1098/rsos.210506>.

</div>

<div id="ref-bershteynEMOD2018" class="csl-entry">

<span class="nocase">Bershteyn, Anna, Jaline Gerardin, Daniel
Bridenbecker, et al.</span> 2018. “Implementation and Applications of
EMOD, an Individual-Based Multi-Disease Modeling Platform.” *Pathogens
and Disease* 76 (5): fty059. <https://doi.org/10.1093/femspd/fty059>.

</div>

<div id="ref-bickerMEmilio2026" class="csl-entry">

Bicker, Julia, Carlotta Gerstein, David Kerkmann, et al. 2026. “MEmilio:
A High Performance Modular EpideMIcs
<span class="nocase">simuLatIOn</span> Software for Multi-Scale and
Comparative Simulations of Infectious Disease Dynamics.” *Scientific
Reports* 16: 26950. <https://doi.org/10.1038/s41598-026-66481-6>.

</div>

<div id="ref-collierRepastHPC2013" class="csl-entry">

Collier, Nicholson, and Michael North. 2013. “Parallel Agent-Based
Simulation with Repast for High Performance Computing.” *SIMULATION* 89
(10): 1215–35. <https://doi.org/10.1177/0037549712462620>.

</div>

<div id="ref-fergusonReport92020" class="csl-entry">

<span class="nocase">Ferguson, Neil M., Daniel Laydon, Gemma
Nedjati-Gilani, et al.</span> 2020. *Report 9: Impact of
Non-Pharmaceutical Interventions (NPIs) to Reduce COVID-19 Mortality and
Healthcare Demand*. Imperial College London.
<https://doi.org/10.25561/77482>.

</div>

<div id="ref-gallagherEpiabm2024" class="csl-entry">

Gallagher, Kit, Ioana Bouros, Nicholas Fan, et al. 2024.
“Epidemiological Agent-Based Modelling Software (Epiabm).” *Journal of
Open Research Software* 12: 3. <https://doi.org/10.5334/jors.449>.

</div>

<div id="ref-hinchOpenABM2021" class="csl-entry">

Hinch, Robert, William J. M. Probert, Anel Nurtay, et al. 2021.
“OpenABM-Covid19—An Agent-Based Model for Non-Pharmaceutical
Interventions Against COVID-19 Including Contact Tracing.” *PLOS
Computational Biology* 17 (7): e1009146.
<https://doi.org/10.1371/journal.pcbi.1009146>.

</div>

<div id="ref-jennessEpiModel2018" class="csl-entry">

Jenness, Samuel M., Steven M. Goodreau, and Martina Morris. 2018.
“EpiModel: An R Package for Mathematical Modeling of Infectious Disease
over Networks.” *Journal of Statistical Software* 84 (8).
<https://doi.org/10.18637/jss.v084.i08>.

</div>

<div id="ref-rossettiNDlib2018" class="csl-entry">

Rossetti, Giulio, Letizia Milli, Salvatore Rinzivillo, Alina Sîrbu, Dino
Pedreschi, and Fosca Giannotti. 2018. “NDlib: A Python Library to Model
and Analyze Diffusion Processes over Complex Networks.” *International
Journal of Data Science and Analytics* 5 (1): 61–79.
<https://doi.org/10.1007/s41060-017-0086-6>.

</div>

<div id="ref-terhoevenMesa2025" class="csl-entry">

<span class="nocase">ter Hoeven, Ewout, Jan Kwakkel, Vincent Hess, et
al.</span> 2025. “Mesa 3: Agent-Based Modeling with Python in 2025.”
*Journal of Open Source Software* 10 (107): 7668.
<https://doi.org/10.21105/joss.07668>.

</div>

</div>
