# Engines


- [epiworld family](#epiworld-family)
- [Covasim](#covasim)
- [Starsim](#starsim)
- [EoN](#eon)
- [epydemic](#epydemic)
- [ixa](#ixa)
- [individual](#individual)
- [FRED](#fred)
- [Agents.jl](#agentsjl)
- [References](#references)

[Back to the project overview](../README.md) · [Methods](methods.md) ·
[Showcase model plan](showcase-models.md)

This document describes the simulation engines the benchmark compares:
what each was built for, how it represents a population and its
dynamics, and what it is typically used for.
[Methods](methods.md#engines) lists how each engine is used in the
benchmark (version, language, time semantics, implementation notes). The
engines are described in the order the benchmark lists them.

## epiworld family

epiworld (Vega Yon 2026) is a header-only C++ library for agent-based
epidemiological models. A model is a population of agents, each in one
of a set of user-defined states, connected by a contact network or by
random mixing. Agents change state through update functions that run on
each step; viruses, tools (such as vaccines or masks), entities (groups
of agents, such as households or schools), and global events (actions
run on a given day) alter the transmission and recovery probabilities of
the agents they touch. The library ships with ready-made models (SIR,
SEIR, SEIRD, and variants with population mixing, quarantine, and
surveillance, among others) and with a general interface for assembling
new ones from states and update functions. Simulations advance in
synchronous daily steps, and replicate runs can be distributed over
threads.

epiworld is designed for speed on populations of millions of agents, and
for exposing the same engine to the languages epidemiologists use.
epiworldR (Meyer and Vega Yon 2023) and epiworldpy (Vega Yon and Banks
2026) wrap the C++ core for R and Python, so a model written in any of
the three runs the same code. The benchmark treats the three as one
engine family and runs all of them, so that it also measures what the
language layer costs.

## Covasim

Covasim (Kerr et al. 2021) is a Python agent-based model written for
COVID-19 and distributed with its parameters. Its population is age
structured, and people contact each other through layers (household,
school, workplace, and community), each with its own contact rate and
transmission multiplier. Each infection follows a native disease course
(exposed, infectious, then asymptomatic, mild, severe, critical,
recovered or dead), with durations and probabilities that depend on age.
Beyond the baseline, Covasim models multiple variants, immunity that
wanes over time, vaccination, testing, contact tracing, and quarantine,
each as a module the user can configure.

Covasim aims at policy analysis: what happens to cases,
hospitalizations, and deaths under a given mix of testing, tracing, and
vaccination. It stores people in arrays and updates them with vectorized
(and, where it helps, compiled) operations. Its development has moved to
its successor, Starsim.

## Starsim

Starsim (Kerr et al. 2025) is a Python framework for agent-based
simulation of disease, built by the team behind Covasim, with a modular
design in which diseases, networks, demographics, and interventions are
separate components that the user combines. It ships with models such as
SIR, SIS, and SEIR and with sexually transmitted infection models, and
it handles several diseases in one simulation, including interactions
between them (coinfection and shared risk factors). Demographic
processes (births, deaths, and pregnancy) and contact structures that
change during the simulation (such as partnership formation and
dissolution on a sexual network) are built in.

Starsim targets questions over long horizons, in which population
turnover and changing contacts matter: how a vaccination or treatment
strategy affects an endemic or slow-moving infection. The benchmark uses
its SEIR disease and a static network built from the shared edge list, a
small part of what the framework offers.

## EoN

EoN (Epidemics on Networks) (Miller and Ting 2019) is a Python package
for epidemic processes on networks represented with NetworkX. It is
built around exact stochastic simulation of continuous-time processes
(Gillespie algorithms), with an event-driven implementation that avoids
simulating steps in which nothing happens. It handles SIR, SIS, and SIRS
dynamics on static networks, with infection and recovery times that need
not be exponential, and simple and complex contagion; it also includes
simulation on temporal networks and discrete-time variants. It also
provides the analytic side of network epidemiology: edge-based
compartmental models and related differential equations that predict the
same quantities without simulating.

EoN is a research tool for studying how network structure shapes
epidemic outcomes, and it is the reference the benchmark uses for a
continuous-time result: it has no simulation clock to discretize. Its
agents are the network’s nodes, and its states are labels on the nodes.

## epydemic

epydemic (Dobson 2022) is a Python library for simulating epidemic
processes on networks, built on NetworkX and the epyc framework for
computational experiments. A model is a set of loci (the nodes or edges
where a process acts) and processes (events with rates, such as
infection along an edge or recovery of a node), assembled into a
compartmented model. The library offers two dynamics for the same model:
a synchronous one, which advances every agent in discrete steps, and a
stochastic one in the style of Gillespie, which draws the next event in
continuous time. Ready-made models include SIR, SIS, and SEIR, and the
process abstraction lets a user add processes that change the network
itself.

epydemic is aimed at experiments: epyc runs a model over a parameter
grid and collects the results, optionally across a cluster. Its
attention is on processes on networks, including networks that rewire or
grow during the epidemic, more than on large populations.

## ixa

ixa (The Ixa Developers 2026) is a Rust framework from the US Centers
for Disease Control and Prevention for building agent-based models. A
model is a context holding a population of people, each with typed
properties (such as an infection status or an age), and a queue of plans
that run at a given simulation time. The simulation executes events from
that queue in time order, so time is continuous and nothing runs on days
when nothing happens. Functionality is organized into modules (for
example, infection, contact, and reporting) that communicate through
events and share the context; ixa provides indexes on properties,
per-module random number streams, and reports.

ixa is built so that a model remains correct and fast as it grows: Rust
provides compiled speed and memory safety, and the module structure
keeps large models maintainable. It does not include epidemic models or
a contact network; a user supplies them, in Rust.

## individual

individual (Charles and Wu 2021) is an R package for individual-based
epidemiological models, written at the MRC Centre for Global Infectious
Disease Analysis at Imperial College London, where it is the engine of
the malaria models of that group. It represents a population with a
small set of objects: states and variables that hold each person’s
categorical, integer, or numeric attributes, events that schedule an
action for a given future time, and processes that run on every time
step. Simulations advance in discrete steps, and the engine applies the
updates queued during a step at its end. The heavy operations (sets of
individuals, scheduling) run in C++ through Rcpp.

individual provides building blocks, not epidemic models: the user
writes the infection process, and it has no notion of contact networks.
It suits models with rich individual heterogeneity (age, immunity,
biting rates) and with other populations that interact with people, such
as the mosquitoes in a vector-borne disease, for which it was designed.

## FRED

FRED (a Framework for Reconstructing Epidemic Dynamics) (Grefenstette et
al. 2013) is an agent-based system from the Public Health Dynamics
Laboratory at the University of Pittsburgh for modeling infectious
disease and its control using census-based synthetic populations. People
belong to places (households, schools, workplaces, neighborhoods, and
others), and an infectious person exposes the other people who share a
place with them each day, with a transmission probability that depends
on the place and on the people. A disease is a set of conditions
(states) with transitions, written in FRED’s own model language, which
also expresses interventions such as vaccination, antivirals, school
closure, and behavior change.

FRED is designed for planning: it has been used for influenza, measles,
and other infections, in populations drawn from census data for
particular US counties. It is a compiled C++ program configured through
input files, not a library called from a general-purpose language, so
the benchmark’s runner writes a model and a population file, runs FRED,
and reads its output.

## Agents.jl

Agents.jl (Datseris et al. 2024) is a Julia package for agent-based
modeling in any field, designed to make models short and fast. A model
has agents of one or more types that live in a space (a graph, a grid,
continuous 2D or 3D space, or an OpenStreetMap road network), a stepping
function that runs for each agent or for the whole model on every step,
and, in recent versions, an event queue for models driven by events. The
space supplies neighbor searches, such as agents within a radius, and
agents can move through it. The package includes tools for collecting
data and for running many replicates in parallel.

Agents.jl is a general-purpose package, so it ships no epidemic model;
the transmission and the contact network in the benchmark are written
for it, and an example epidemic in its documentation is the starting
point for this kind of model. It is at its strongest where the contact
structure comes from space: agents that move, meet others nearby, and
are affected by where they are.

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
Flexible and General Agent-Based Model Engine*. V. 0.17.1. Released.
<https://github.com/UofUEpiBio/epiworld>.

</div>

<div id="ref-vegayonEpiworldpy2026" class="csl-entry">

Vega Yon, George G., and Olivia Banks. 2026.
*<span class="nocase">epiworldpy</span>: Python Bindings for Epiworld*.
V. 0.17.1-0. Released. <https://github.com/UofUEpiBio/epiworldpy>.

</div>

</div>
