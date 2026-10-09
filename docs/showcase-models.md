# Showcase models: a plan

[Back to the project overview](../README.md) ·
[Engines](engines.md) ·
[Methods](methods.md)

This is the plan for [issue #24](https://github.com/UofUEpiBio/epiworld-benchmark/issues/24),
*Find scenarios where the other engines shine*. It defines, for each engine,
the model it is best suited to, so that the benchmark can later implement
those models. Nothing here is implemented yet, and every statement about what
an engine can or cannot do is an initial assessment to be verified in step 2
of the [workflow](#workflow).

## Why

Scenarios 00 to 04 are one model, SEIRH on a contact network, made larger,
more complex, or more realistic. That model suits epiworld, which the
benchmark's authors know best, and it leaves out what several other engines
were built for: co-location (agents meeting because they share a place or
are close in space), age-structured layers, demographic turnover, continuous
time with non-exponential durations, networks that change with the epidemic,
and models without a network at all. A comparison on only the shared model
says how fast each engine runs the model it was not designed for. A fair
comparison also gives each engine a model it was designed for, and then
reports honestly which other engines can express it, how, and at what cost.

## Principles

1. **One showcase model per engine.** Each is the model that most
   shows what its engine does well, specified in terms of states, parameters,
   and outputs rather than any engine's API, so that any engine can try it.
   The epiworld family has one showcase for its three runners.
2. **Every engine attempts every showcase.** A runner uses the engine's own
   way of expressing the model, as [scenarios.md](../scenarios.md#checklist)
   already requires. What differs between engines is how much of it is
   built in (**native**), how much must be written for the benchmark
   (**written**), and what cannot be expressed (**unsupported**).
3. **An unsupported cell is a result.** When an engine cannot implement a
   showcase, the report says which component is missing and why, rather than
   dropping the engine silently or approximating the model without saying so.
4. **Compare engines within a showcase, not across showcases.** Different
   models have different run times by construction. The tables compare
   engines on the same showcase, and show how each engine's cost on its own
   showcase compares with its cost on the baseline (scenario 00), which is
   the fair summary of whether a feature costs more or less than the generic
   model.
5. **Keep the measurement protocol.** Same container, seeds, sequential
   replicates, timings, memory, and code-size counts as in
   [Methods](methods.md#execution-protocol). New inputs (a population with
   ages and places, positions in space) are generated once, hashed into each
   record's fingerprint, and shared, the way the edge list is.
6. **Keep the output contract where it applies.** Every showcase reports the
   five final compartments (summing to `n`) and `peak_hospitalized`, so the
   epidemiological agreement checks of the existing reports still work. A
   model with other states (births, deaths, vector infection) folds them into
   those five and documents the convention, and adds its own outputs next to
   them. Where a model has no hospitalization branch (a malaria model, for
   example), its report says so and the field is zero.

## The showcase models

Scenario numbers continue from the current last scenario (04) in the order of
the table, and may change as the models are refined.

| Scenario | Showcase of | Model | Component the engine is built for |
|:---------|:------------|:------|:----------------------------------|
| 05 | epiworld family | SEIR with three age groups, a contact matrix, and quarantine of detected cases | Population mixing between groups, with isolation as a built-in model || W 
| 06 | Covasim | Age-structured COVID-like disease on household, school, workplace, and community layers, with test-and-trace | Multi-layer contacts, age-dependent severity, testing and tracing || W 
| 07 | Starsim | SIS infection on a dynamic partnership network with births and deaths | Demographics and contacts that form and dissolve || W 
| 08 | EoN | SIR in continuous time on a weighted network with Gamma-distributed infectious periods | Exact event-driven simulation with non-exponential durations || N 
| 09 | epydemic | SIR on an adaptive network in which susceptible agents rewire away from infectious ones | Processes that change the network || W 
| 10 | ixa | Continuous-time disease with an infectiousness profile, individual heterogeneity, and delayed isolation | Plans on an event queue and person properties || W 
| 11 | individual | Host–vector malaria-like transmission with age-dependent biting | Two interacting populations, no contact network || ? 
| 12 | FRED | Influenza in a synthetic population with household, school, and workplace places and reactive school closure | Place-based co-location and a policy language || W 
| 13 | Agents.jl | Mobile agents in continuous space with proximity transmission and a movement restriction | Spatial agents, co-location by distance || ? 
| 14 | ABM | SEIR in continuous time in an open population, with births, deaths, and Gamma-distributed periods | Event-driven agents that join and leave, with arbitrary waiting times |
| 15 | EpiModel | SI infection on a partnership network drawn each day from a temporal ERGM fitted to degree, mixing, and duration targets | Networks as statistical models: partnerships that form and dissolve to match observed statistics |

The sections below specify each model. Parameters are starting points, to be
settled when the model is implemented; the baseline's disease values (a mean
latent period of 4 days, a mean infectious period of 7 days, $R_0$ of 2,
100 seed cases) apply wherever a model has the same state.

### 05. epiworld family: mixing and quarantine

- **Model.** SEIR in three age groups (children, adults, older adults) with
  random mixing weighted by a contact matrix (an assortative matrix, with
  most contacts inside a group). Infectious agents are detected with a
  probability per day, and detected agents and their contacts are
  quarantined for a fixed period, during which their contacts are reduced.
- **Parameters.** Group sizes (for example 0.2, 0.6, 0.2 of the population),
  the contact matrix, detection probability, quarantine duration, contact
  reduction in quarantine, and the baseline disease parameters.
- **Outputs.** The baseline outputs, the final compartments by group, and the
  number of agents quarantined.
- **Why it suits epiworld.** It is epiworld's built-in mixing model with
  quarantine, a model of its own library rather than a runner assembled from
  states, and it exercises entities and global events.
- **To verify.** That the epiworldR and epiworldpy wrappers expose the model's
  parameters; the benchmark's runners use only what the wrappers provide.

### 06. Covasim: layered contacts and test-and-trace

- **Model.** Age-structured population with four contact layers
  (household, school, workplace, community) and a native COVID-like course:
  exposed, infectious, then asymptomatic, mild, severe (hospitalized), or
  critical, recovered or dead. Susceptibility and severity depend on age.
  Symptomatic people are tested with a probability per day, and positives are
  isolated while their contacts are traced and quarantined with a
  probability and a delay.
- **Parameters.** Age distribution, layer contact rates and transmission
  multipliers, age-dependent severity, testing probability and delay, test
  sensitivity, tracing probability and delay, and isolation period.
- **Outputs.** The baseline outputs, plus deaths, tests, and people traced
  and quarantined, with deaths reported in their own column (and counted in
  the recovered compartment in the five-compartment contract).
- **Why it suits Covasim.** Layers, age structure, the severity cascade, and
  testing and tracing are Covasim's native features and the reason it exists.
- **To verify.** The layer structure Covasim generates by default (its
  synthetic population) compared with a population file shared with every
  engine; the benchmark may need to load Covasim's layers from the shared file.

### 07. Starsim: partnerships and demography

- **Model.** SIS (susceptible and infected, with recovery or treatment
  returning to susceptible) on a sexual-partnership network in which
  partnerships form and dissolve over the simulation, in a population with
  births and deaths, simulated over years rather than days.
- **Parameters.** Partnership formation and duration, concurrency, birth and
  death rates by age, per-partnership transmission, and a treatment or
  recovery rate. A later extension adds a second infection with an
  interaction (coinfection raising susceptibility).
- **Outputs.** Prevalence over time, the final compartments, and the
  population size at the end, which now differs from `n` because of
  turnover (the report documents how the contract's sum is adapted).
- **Why it suits Starsim.** Demographics, dynamic networks, and multiple
  interacting diseases are its design center.
- **To verify.** How each other engine represents a growing and shrinking
  population (most fix the number of agents).

### 08. EoN: continuous time with general durations

- **Model.** SIR in continuous time on a weighted network. Infection along an
  edge is a Poisson process with rate proportional to the edge weight, and
  the infectious period is Gamma distributed (a shape other than 1 makes it
  non-Markovian). Outputs are exact event times.
- **Parameters.** The edge weights (for example, drawn from a log-normal
  distribution), the Gamma shape and scale (keeping the mean infectious
  period of 7 days), and the transmission rate calibrated to $R_0$.
- **Outputs.** The baseline outputs and the epidemic's final size, plus a
  comparison of the simulated mean final size with the analytic prediction
  EoN provides, a check no other engine offers.
- **Why it suits EoN.** Its event-driven Gillespie algorithms handle
  non-exponential durations exactly, and its analytic models give the answer
  without simulating.
- **To verify.** That the synchronous engines can approximate Gamma durations
  by sampling durations on entry (Starsim does) and by what error.

### 09. epydemic: an adaptive network

- **Model.** SIR in which every edge between a susceptible and an infectious
  agent is dropped at a rate and replaced by an edge from the susceptible
  agent to a random susceptible agent (the adaptive-rewiring model),
  so the epidemic changes the network it spreads on.
- **Parameters.** The rewiring rate or probability, the baseline transmission
  and recovery parameters, and the initial network (the shared
  Watts–Strogatz graph).
- **Outputs.** The baseline outputs, plus the count of edges by type of
  endpoints (susceptible–susceptible, susceptible–infectious,
  infectious–infectious, and so on) over time, and the number of
  rewirings.
- **Why it suits epydemic.** Network-changing processes are expressed as
  processes in its model, and its dynamics handle them in either discrete or
  continuous time.
- **To verify.** How the other engines express a mutable network, which is
  expected to be an adjacency structure rewritten by hand in most of them.

### 10. ixa: event-driven infectiousness profiles

- **Model.** A person's infectiousness follows a profile over time since
  infection (the Gamma-distributed generation interval of a respiratory
  disease), scaled by an individual relative infectiousness drawn from a
  heavy-tailed distribution (superspreading). Symptomatic cases are isolated
  after a delay, and the isolation is a plan on the event queue that removes
  their contacts.
- **Parameters.** The generation interval's shape and scale, the dispersion
  of individual infectiousness, the probability and delay of isolation, and
  the isolation's effectiveness.
- **Outputs.** The baseline outputs, plus the realized distribution of
  secondary cases per case and the mean generation interval.
- **Why it suits ixa.** Plans in continuous time and typed person
  properties express each of these components directly, and nothing runs when
  nothing happens.
- **To verify.** How the daily-step engines represent a time-varying
  infectiousness (state durations, or a per-agent counter written by hand).

### 11. individual: hosts and vectors

- **Model.** A host–vector model with no contact network: humans (SEIS, with
  partial protection after infection) and mosquitoes (susceptible, exposed,
  infectious) infect each other through bites. Human biting rates depend on
  age and vary between individuals, and a bed-net intervention reduces them
  for a fraction of the population.
- **Parameters.** The number of mosquitoes per human, the biting rate,
  transmission probabilities in both directions, latent periods, a
  mosquito mortality rate, age-dependent biting, and net coverage and
  efficacy.
- **Outputs.** Human prevalence over time and the final compartments of the
  humans, with the mosquitoes as extra columns. There is no hospitalization
  branch.
- **Why it suits individual.** Variables for individual attributes, events for
  delayed transitions, and processes over several populations are its
  building blocks, and it is the engine behind a large malaria modeling
  codebase.
- **To verify.** That the network-centric engines can host a second
  population that is not on the network (epiworld's entities or a separate
  agent set, EoN and epydemic probably not).

### 12. FRED: places and a policy language

- **Model.** Influenza in a synthetic population whose people belong to a
  household, a school or workplace, and the community. Infectious people
  infect others who share a place each day; transmission in schools is
  higher than in households. A reactive school closure starts when the
  prevalence among students passes a threshold, and ends after a fixed
  period.
- **Parameters.** The synthetic population (the benchmark's GeoPops-derived
  population, kept as places and not only as a collapsed network), place-type
  transmission probabilities, the closure threshold and duration, and the
  baseline disease parameters.
- **Outputs.** The baseline outputs, plus the infections by place type and
  the days schools were closed.
- **Why it suits FRED.** Place-based exposure and a language for conditions
  and policies are its design. This is the co-location the issue describes.
- **To verify.** Which engines can represent places natively (Covasim's layers,
  Starsim's mixing pools, and epiworld's entities are candidates) and which
  must expand places into pairs of agents. Scenario 04 collapses places into
  edges and so cannot show this.

### 13. Agents.jl: movement in space

- **Model.** Agents move at random in a continuous two-dimensional space
  with periodic boundaries. A susceptible agent within a radius of an
  infectious one is infected with a probability per step, so contacts arise
  from where agents are and not from a fixed graph. A movement restriction
  reduces the speed of a fraction of agents after prevalence passes a
  threshold. States are SEIRH as in the baseline.
- **Parameters.** The size of the space and the population density (for the
  same mean contacts per agent as the baseline, about 10), the step length,
  the infection radius, the restricted fraction and its speed reduction, and
  the baseline disease parameters.
- **Outputs.** The baseline outputs, plus the mean contacts per agent per
  day.
- **Why it suits Agents.jl.** Space and neighbor searches are what the
  package provides; the model is a few lines.
- **To verify.** Which other engines can place agents in space. Most are
  expected to need a hand-written neighbor search, or to be unable to express
  it.

### 14. ABM: an open population in continuous time

- **Model.** SEIR in continuous time on a network in which agents die at a
  rate and are replaced by susceptible newborns who connect to a few random
  agents, so the population turns over during the epidemic. The latent and
  infectious periods are Gamma distributed.
- **Parameters.** The birth and death rate (a mean lifetime much longer than
  the epidemic, and one comparable to it), the number of connections of a
  newborn, the Gamma shapes and scales (keeping the baseline means of 4 and
  7 days), and the transmission rate calibrated to $R_0$.
- **Outputs.** The baseline outputs, plus the population size at the end of
  each day. Because $n$ changes, the five compartments sum to the final
  population and not to $n$. There is no hospitalization branch.
- **Why it suits ABM.** Its events are ordered in continuous time, agents can
  leave and join a population while the simulation runs, and a waiting time
  can be any distribution, so the model needs no discretisation.
- **To verify.** Whether the discrete-time engines can represent turnover on
  a network (Starsim has demographics; the others would add and remove agents
  by hand or fix the population), and whether EoN can handle a network that
  changes.

### 15. EpiModel: a fitted temporal network

- **Model.** An HIV-like SI infection (no recovery) on a partnership network
  that is not fixed but drawn every day from a separable temporal
  exponential random graph model (STERGM). The formation model targets the
  mean degree, the share of agents with more than one partner (concurrency),
  and mixing by a binary attribute; the dissolution model targets a mean
  partnership duration. Simulated over years, with a closed population at
  first and births and deaths as an extension.
- **Parameters.** The target statistics and the mean duration, a
  per-act transmission probability and the acts per partnership per day,
  and the attribute's distribution. The fit (`netest()`) and its
  diagnostics (`netdx()`) are part of the model.
- **Outputs.** Prevalence over time, the final compartments, and the
  network's statistics over the run against their targets, which show that
  the network the epidemic ran on is the one the model describes.
- **Why it suits EpiModel.** It was built for exactly this: an epidemic on a
  network whose structure and turnover are estimated from partnership data.
  The fitted network is the part the benchmark's fixed edge list turns off.
- **To verify.** Whether any other engine can draw a network that matches
  target statistics (Starsim's dynamic partnership networks come closest),
  or whether the others must replay a sequence of networks that EpiModel
  writes out, which would compare the epidemics but not the network model.

## Expected feasibility

Initial assessment of how each engine would implement each showcase, to be
checked in step 2 of the workflow. **N** means built in or close to it, **W**
means expressible with code written for the benchmark, **?** means unclear,
and **X** means likely unsupported. Rows are the showcase models, and columns
are the engines (epiworld = the epiworld family).

| Showcase | epiworld | Covasim | Starsim | EoN | epydemic | ixa | individual | ABM | EpiModel | FRED | Agents.jl |
|:---------|:--------:|:-------:|:-------:|:---:|:--------:|:---:|:----------:|:---:|:--------:|:----:|:---------:|
| 05 mixing and quarantine | N | W | W | X | W | W | W | W | W | W | W |
| 06 layers and test-and-trace | W | N | W | X | X | W | W | W | W | W | W |
| 07 partnerships and demography | W | W | N | X | ? | W | W | W | N | W | W |
| 08 continuous time, Gamma durations | X | W | W | N | W | N | W | N | X | W | W |
| 09 adaptive network | ? | X | W | ? | N | W | W | W | W | X | W |
| 10 infectiousness profile and isolation | ? | W | W | ? | ? | N | W | W | W | W | W |
| 11 host–vector | W | ? | W | X | X | W | N | ? | ? | ? | W |
| 12 places and school closure | W | W | W | X | X | W | W | W | W | N | W |
| 13 movement in space | X | X | ? | X | X | W | W | ? | X | ? | N |
| 14 open population, continuous time | W | W | N | X | ? | W | W | N | W | W | W |
| 15 fitted temporal network | X | X | W | X | ? | W | W | W | N | X | W |

The matrix is a hypothesis. Its purpose is to say where the work is: the
cells marked **X** or **?** are those that need a decision before
implementation, for instance whether an engine's model is merely not shipped
(so **W**) or its design rules it out (so **X**).

## Workflow

For each showcase, in this order, as the issue lays out:

1. **Specify.** Write the model's states, transitions, parameters, inputs, and
   outputs in the scenario's `scenario.toml` and report, in engine-neutral
   terms, with the reference engine's implementation first.
2. **Verify feasibility.** For every other engine, work out whether it can
   express each component, natively, by hand, or not at all, from its
   documentation and a prototype. Record the result as the engine's
   **support level** in `scenario.toml` with a one-line reason for each
   component that is not native.
3. **Implement and run.** Write the runner for each engine that can express the
   model, using the engine's own tools, and run the scenario under the usual
   protocol.
4. **Document the gaps.** For each engine that cannot, the scenario's report
   states the missing component and the reason, in a table next to the
   results.
5. **Check agreement.** Compare epidemiological outcomes across the engines
   that ran, as the existing reports do. Calibrate only where the engines'
   semantics differ and say so, as in [Calibration](methods.md#calibration).

### Changes to the benchmark that the plan needs

- **Shared inputs beyond an edge list.** Showcases 05, 06, 12, and 13 need a
  population file (ages, groups, places, or positions) generated once and
  shared, like the network. This needs a generator in `scripts/` and the
  file's hash in the fingerprint, as the network has.
- **Unsupported engines in `scenario.toml`.** An `[unsupported]` table that
  maps an engine to its reason, so that `run.py` skips the engine instead of
  failing, and the report prints the reasons. A scenario's `engines` list can
  already restrict it to a subset, but it has nowhere to say why.
- **A relaxed record contract.** The checks that the five compartments sum to
  `n` need to allow the demographic showcases (07 and 14), where `n` changes, and
  the models without a hospitalization branch (11 and 14).
- **Reports.** Each scenario's report keeps its current structure and adds a
  support table; the overview's engine table gains a column for the engine's
  showcase.

## Order of work

1. **Contract and plumbing.** The `[unsupported]` table, the relaxed record
   contract, and the shared population generator, with tests. No new model.
2. **Showcases that reuse the network**: 08 (EoN), 09 (epydemic), and 10 (ixa)
   run on the existing edge list, and are the cheapest to start with.
3. **Showcases that need a population file**: 05 (epiworld), 06 (Covasim),
   and 12 (FRED).
4. **Showcases with their own world**: 07 (Starsim), 11 (individual),
   13 (Agents.jl), and 14 (ABM).

Each showcase is its own pull request, so each can be reviewed and its
results published on their own.

## Open questions

- Should showcases be in the main overview's summary figures, or reported
  separately, since their run times are not comparable with the baseline's?
- Is the reference engine's implementation (the engine the showcase is for)
  to be reviewed by someone who maintains or knows that engine, to avoid
  writing a model that engine's users would call unidiomatic?
- Is a showcase that only one engine can run (13, for example) still part of
  the benchmark, or is it a case study? The plan assumes it is part of the
  benchmark, with the others' results as `unsupported`.
- Which version of each engine pinned in the image has the features each
  showcase needs? Step 2 checks this, and a showcase may motivate a version
  bump or a patch, as epiworldpy's `UpdateFun` factories did.
