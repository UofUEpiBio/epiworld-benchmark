# epiworld and ixa: where the time goes

[Back to the project overview](README.md)

epiworld (in C++, R, or Python) and ixa are the two fastest engines in every
scenario. This note measures where each spends its time, to explain how the
two compare. The explanation is the same in every scenario; the last section
maps it onto each scenario's results.

In short:

- **Per replicate, epiworld is faster from 100,000 agents up.** The benchmark
  times what each engine has to redo for another replicate on the same
  network (see [what is measured](README.md#run-time)). For ixa that includes
  building a new context, the population, network, and index, because
  `execute()` runs a context only once. That takes about 26 ms at 100,000
  agents and 360 ms at 1,000,000. epiworld builds its model once and resets it
  at the start of every `run()`, which takes 1.2 ms and 15 ms.
- **The simulation proper is faster in ixa.** Comparing ixa's `execute()`
  alone with epiworld's `run()`, ixa is faster at every size. The two spend
  about the same time on the epidemic itself. The difference comes from two
  costs in epiworld that do not depend on it:
  1. **Reading the transmission probability by name**, once for every
     susceptible neighbour of every infectious agent. This is about 40% of
     epiworld's time at 10,000 and 100,000 agents.
  2. **Re-initializing the population at the start of every run**, which
     grows with the population rather than the outbreak. It is what makes the
     model reusable, and it is far cheaper than ixa's rebuild, but ixa's
     `execute()` does not pay it.

  In the scenarios with a vaccine, placing the vaccine is a third, smaller
  cost.

## How it was measured

These are side measurements, not benchmark results. They time the
scenarios' own runners, one process at a time as the benchmark does, in the
benchmark's container, on the benchmark's networks and seeds. [`analysis/profile.sh`](analysis/profile.sh) reproduces
them:

```sh
make container-profile
```

It times each runner over nine seeds, and reports the median:

- **As benchmarked.** For ixa this includes building the context.
- **ixa's `execute()` alone.** The same ixa runner with the timer around
  `execute()` only, as the benchmark timed it before it counted building the
  context as simulation.
- **epiworld with a constant probability.** The C++ runner with
  `pathogen.set_prob_infecting(beta)` in place of
  `pathogen.set_prob_infecting("Transmission rate")`. The model and its
  random numbers are unchanged, so the epidemics are identical.
- **Each of these with no transmission** (`--transmission-multiplier 0`). The
  100 seed cases still progress, but nobody else is infected. What is left is
  the fixed cost of a run: whatever the engine does regardless of the
  outbreak.

It then runs the C++ runner once per size against a copy of epiworld's
headers with a timer around each phase of `Model::run()`
([`analysis/instrument_epiworld.py`](analysis/instrument_epiworld.py)).

## Scenario 00, by size

Median time in milliseconds:

| Agents | epiworld `run()` | epiworld, constant probability | ixa `execute()` | ixa, per replicate | epiworld, no transmission | ixa `execute()`, no transmission | ixa, building the context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 9.3 | 5.3 | 4.6 | 7.2 | 0.53 | 0.15 | 2.6 |
| 100,000 | 14.7 | 9.1 | 8.4 | 34.7 | 1.8 | 0.24 | 26 |
| 1,000,000 | 34.1 | 27.1 | 13.6 | 371 | 17.3 | 0.62 | 360 |

The last column is ixa per replicate minus `execute()`, both with no
transmission. Subtracting each engine's fixed cost from its time with
transmission leaves the time spent on the epidemic: 5.0, 7.6, and 8.1 ms for
epiworld with a constant probability, and 4.5, 8.1, and 13.0 ms for ixa's
`execute()`. On the outbreak itself the two are level.

### What ixa redoes for every replicate

ixa's `execute()` runs the plans in a context until none are left, and the
context cannot be reset. Another replicate on the same network therefore
needs a new context: adding one entity per agent, two edges per contact, and
the index on disease status, which `add_entity()` keeps up to date as agents
are added. That is 2.6 ms at 10,000 agents, 26 ms at 100,000, and 360 ms at
1,000,000, and it is most of ixa's time per replicate from 100,000 agents up.

epiworld's equivalent is `reset()`, below. It keeps the network and the
agents, and only returns them to their initial state, so it is 20 to 25 times
cheaper than ixa's rebuild at every size. A user running many replicates, or
changing parameters between runs, pays it instead of rebuilding the model;
`run_multiple()` relies on it.

### 1. The transmission probability is looked up by name

`Virus::set_prob_infecting("Transmission rate")` stores a closure that calls
`Model::get_param(std::string)`. Push transmission calls it for every
susceptible neighbour of every infectious agent, hundreds of thousands of
times per run here. Each call copies the name, which at 17 characters is too
long for the string's inline buffer and so allocates, and then searches the
parameter map twice (`find()`, then `operator[]`). The value depends only on
the infector's virus, and here it never changes.

Setting the probability as a number removes 4.0, 5.6, and 7.0 ms per run:
43% of the C++ runner's time at 10,000 agents, 38% at 100,000, and 21% at
1,000,000. At 100,000 agents the instrumented run puts push transmission at
11.2 ms of `run()`'s 16.4, so the lookup is about half of it.

epiworldR and epiworldpy go through the same call: epiworldR's
`set_prob_infecting_ptr()`, despite its name, and epiworldpy's
`set_prob_infecting()` both pass the parameter's name to it. Resolving the
parameter once, when the run starts, would give every wrapper the
constant-probability times without any change to the models users write.

### 2. `run()` re-initializes the population

`Model::run()` starts with `reset()`, which walks the whole population
several times before the first day. It resets each agent, rebuilds the index
of agents by state, recounts the database's totals, clears the queue, and,
to draw the 100 seed cases, builds a list of every agent without a virus:

| Phase of `reset()` | 10,000 agents (ms) | 100,000 agents (ms) | 1,000,000 agents (ms) |
|:---|---:|---:|---:|
| Reset each agent | 0.02 | 0.18 | 2.5 |
| Rebuild the state index | 0.04 | 0.35 | 5.2 |
| Reset the database | 0.01 | 0.12 | 1.5 |
| Clear the queue | 0.01 | 0.13 | 1.6 |
| Seed the initial cases | 0.05 | 0.42 | 4.1 |
| **Total** | **0.13** | **1.2** | **15.0** |

The runs with no transmission show the same from the outside: epiworld's
fixed cost grows with the population (0.53, 1.8, and 17.3 ms), while that of
ixa's `execute()` barely does (0.15, 0.24, and 0.62 ms), because its context
arrives already built.

`reset()` is small next to the epidemic at 10,000 and 100,000 agents. At
1,000,000 agents, with an outbreak of about 6,000 cases, it is nearly half
of `run()`. Several of its passes could be merged or skipped: on a model's first
run the agents are already in their initial state, and drawing 100 seed
cases does not need a list of a million candidates.

### What is not part of the gap

- **Building the network.** Earlier versions of this analysis traced about
  11 ms of the C++ runner's simulation time at 100,000 agents to
  `agents_from_edgelist()`. It built a temporary adjacency list from about a
  million small allocations, and freeing them left glibc's allocator to
  consolidate them during the next large allocation, which happened inside
  `run()`. epiworld now builds the network with a counting sort
  ([UofUEpiBio/epiworld#274](https://github.com/UofUEpiBio/epiworld/issues/274)),
  and the benchmark pins a commit that includes it. The charge is gone, and
  with it the C++ runner's lag behind epiworldR and epiworldpy that earlier
  reports showed at 100,000 agents.
- **Bookkeeping.** epiworld records the transmission tree, the transition
  matrix, and daily counts in every run. Recording a day in the database
  (`next()`) takes about 0.2 µs, and applying the day's events 0.7 ms per run
  at 100,000 agents. [Scenario 02](scenario_02/README.md), where every engine
  has to record the same outputs, confirms it: ixa adds that bookkeeping at
  little cost, and the gap does not change.
- **The daily loop's overhead.** In the runs with no transmission, 100 days
  add almost nothing to `reset()`: epiworld's daily work follows the agents
  who are infected, as ixa's does.

## Scenarios 01 and 02: the vaccine

Every runner times the vaccine's distribution as part of the simulation, as
it times seeding, because epiworld places its tools inside `run()`. Earlier
versions of the benchmark timed it only for some engines; the comparison is
now like for like.

epiworld places the vaccine on about 24,000 agents at 100,000 agents in
`reset()`, and each placement clones the tool onto the heap and queues an
event. The instrumented run spends 3.6 ms placing tools and seeds, against
0.42 ms for the seeds alone in scenario 00: about 3.2 ms for the vaccine. ixa
sets a property on its 30,000 vaccinees in a plan at time 0, which raises the
fixed cost of `execute()` from 0.24 to 1.9 ms, about 1.7 ms for the vaccine.

Median time in milliseconds, scenario 01's model:

| Agents | epiworld `run()` | epiworld, constant probability | ixa `execute()` | ixa, per replicate | epiworld, no transmission | ixa `execute()`, no transmission |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 4.4 | 2.6 | 1.9 | 4.2 | 0.86 | 0.31 |
| 100,000 | 9.5 | 8.3 | 4.2 | 30.6 | 5.4 | 1.9 |

The heap also matters here. The tool clones are the only large batch of
small allocations inside `run()`, and a fresh C++ process has to grow its
heap to serve them, while R and Python start with a large one. This is why
the C++ runner is slower than epiworldR and epiworldpy in scenarios 01 and
02. Letting glibc keep a large heap in reserve (`MALLOC_TOP_PAD_=268435456`)
brings the C++ runner from 10.9 to 9.2 ms at 100,000 agents, over 15 seeds;
serving large blocks from the heap (`MALLOC_MMAP_THRESHOLD_=1073741824`)
brings it to 9.0 ms. Sharing one tool object among agents would remove these
allocations altogether.

The vaccine shrinks the outbreak, so the parameter lookup matters less here
(1.2 to 1.8 ms), and the fixed cost, with the vaccine, matters more: at
100,000 agents it is more than half of `run()`. ixa's rebuild still
dominates its time per replicate. Scenario 02 runs the same model and the
same epidemics, so the same holds there.

## In each scenario's results

Median simulation time per replicate in the benchmark itself, from each
scenario's report. Like the side measurements, the benchmark runs one
replicate at a time:

| Scenario | Agents | epiworld (C++) | epiworldR | epiworldpy | ixa | ixa / epiworld (C++) |
|:---|---:|---:|---:|---:|---:|---:|
| [00](scenario_00/README.md) | 10,000 | 8.8 | 9.0 | 8.3 | 7.2 | 0.82 |
| [00](scenario_00/README.md) | 100,000 | 15.2 | 15.0 | 14.0 | 38.5 | 2.53 |
| [01](scenario_01/README.md) | 10,000 | 4.3 | 4.0 | 3.9 | 4.3 | 1.00 |
| [01](scenario_01/README.md) | 100,000 | 11.7 | 9.0 | 9.5 | 33.3 | 2.85 |
| [02](scenario_02/README.md) | 10,000 | 4.8 | 4.0 | 3.9 | 4.7 | 0.97 |
| [02](scenario_02/README.md) | 100,000 | 10.6 | 9.0 | 8.7 | 32.7 | 3.10 |
| [03](scenario_03/README.md) | 1,000,000 | 35.8 | 33.0 | 39.6 | 460.2 | 12.84 |

Times are in milliseconds.

- **At 10,000 agents ixa is a little faster.** Its rebuild is small here
  (about 2.6 ms), and costs less than the parameter lookup and `reset()` cost
  epiworld: ixa takes 0.82 times as long as the C++ runner in scenario 00.
  With the vaccine the two are level.
- **At 100,000 agents epiworld is about three times faster per replicate.**
  Rebuilding ixa's context now takes more time than the whole of epiworld's
  `run()`. The ratio is largest in scenarios 01 and 02 (2.8 and 3.1 against
  2.5), where the vaccine shrinks the outbreak but not the rebuild.
- **At 1,000,000 agents ([scenario 03](scenario_03/README.md)) the rebuild
  dominates.** ixa takes about 13 times as long as epiworld per replicate,
  although its `execute()` alone is still faster than epiworld's `run()`, of
  which `reset()` is now nearly half.

## What would make epiworld faster

These would be changes to epiworld, not to the benchmark's runners, which
use each engine as its documentation shows:

- Resolve a virus's named parameters once, when the run starts, instead of
  on every call. This alone would bring `run()` close to ixa's `execute()` at
  10,000 and 100,000 agents in scenario 00.
- Make `reset()` cheaper for large populations: merge its passes over the
  agents, skip resetting agents that are already in their initial state, and
  draw the seed cases without first listing every candidate.
- Share one tool object among the agents that receive it, rather than
  cloning it for each.
