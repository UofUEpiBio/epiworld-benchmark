# epiworld and ixa: where the time goes

[Back to the project overview](../README.md)

The measurements below were taken at epiworld `a1ff20a` (0.17.0). The
benchmark now pins 0.17.1, which revised how epiworld chooses between pushing
and pulling transmission
([UofUEpiBio/epiworld#281](https://github.com/UofUEpiBio/epiworld/pull/281))
and sped up the pull scan; this note has not been remeasured since.

epiworld (in C++, R, or Python) and ixa are the two fastest engines in every
scenario. This note measures where each spends its time, to explain how the
two compare. The explanation is the same in every scenario; the last section
maps it onto each scenario's results.

In short:

- **Per replicate, epiworld is now faster at every size**, including 10,000
  agents, where ixa used to be a little faster. The benchmark times what each
  engine has to redo for another replicate on the same network (see [what is
  measured](methods.md#timing)). For ixa that includes building a new
  context, the population, network, and index, because `execute()` runs a
  context only once. That takes about 2.6 ms at 10,000 agents, 29 ms at
  100,000, and 374 ms at 1,000,000. epiworld builds its model once and resets
  it at the start of every `run()`, which takes 0.17, 1.4, and 17 ms.
- **On the epidemic computation itself, the two are close**, and epiworld now
  has a small edge from 100,000 agents up. Comparing ixa's `execute()` alone
  with epiworld's `run()` with a constant transmission probability, each
  minus its own fixed cost (its time with no transmission), the two are level
  at 10,000 agents; epiworld is faster at 100,000 and 1,000,000. This is a
  change from earlier versions of this analysis, where ixa's `execute()` was
  faster at every size:
  1. **Reading the transmission probability by name no longer costs much.**
     epiworld resolves a virus's named parameters once per model, not on
     every call (see below); the gap between epiworld as benchmarked and
     epiworld with the same probability set as a plain number, which used to
     be 21-43% of `run()`, is now within run-to-run noise at every size.
  2. **Re-initializing the population at the start of every run** remains a
     real cost, and it grows with the population rather than the outbreak.
     It is what makes the model reusable, and it is still far cheaper than
     ixa's rebuild, but ixa's `execute()` does not pay it.

  In the scenarios with a vaccine, placing the vaccine is a third, smaller
  cost.

## How it was measured

These are side measurements, not benchmark results. They time the
scenarios' own runners, one process at a time as the benchmark does, in the
benchmark's container, on the benchmark's networks and seeds. [`analysis/profile.sh`](../analysis/profile.sh) reproduces
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
([`analysis/instrument_epiworld.py`](../analysis/instrument_epiworld.py)).

## Scenario 00, by size

Median time in milliseconds:

| Agents | epiworld `run()` | epiworld, constant probability | ixa `execute()` | ixa, per replicate | epiworld, no transmission | ixa `execute()`, no transmission | ixa, building the context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 5.02 | 4.63 | 4.72 | 7.31 | 0.31 | 0.16 | 2.6 |
| 100,000 | 8.09 | 8.49 | 9.47 | 37.75 | 1.84 | 0.27 | 28.6 |
| 1,000,000 | 26.65 | 25.57 | 11.63 | 401.83 | 17.61 | 0.62 | 374.3 |

The last column is ixa per replicate minus `execute()`, both with no
transmission. Subtracting each variant's own fixed cost from its time with
transmission leaves the time spent on the epidemic: 4.4, 6.8, and 9.2 ms for
epiworld with a constant probability, and 4.6, 9.2, and 11.0 ms for ixa's
`execute()`. The two are level at 10,000 agents; epiworld is faster at
100,000 and 1,000,000.

### What ixa redoes for every replicate

ixa's `execute()` runs the plans in a context until none are left, and the
context cannot be reset. Another replicate on the same network therefore
needs a new context: adding one entity per agent, two edges per contact, and
the index on disease status, which `add_entity()` keeps up to date as agents
are added. That is 2.6 ms at 10,000 agents, 28.6 ms at 100,000, and 374 ms at
1,000,000, and it is most of ixa's time per replicate from 100,000 agents up.

epiworld's equivalent is `reset()`, below. It keeps the network and the
agents, and only returns them to their initial state, so it is 15 to 23 times
cheaper than ixa's rebuild at every size. A user running many replicates, or
changing parameters between runs, pays it instead of rebuilding the model;
`run_multiple()` relies on it.

### 1. The transmission probability is looked up by name (now resolved once)

Earlier versions of this analysis found that
`Virus::set_prob_infecting("Transmission rate")` stored a closure that called
`Model::get_param(std::string)` on every push-transmission check, for every
susceptible neighbour of every infectious agent, hundreds of thousands of
times per run. Each call copied the name, which at 17 characters is too long
for the string's inline buffer and so allocated, and then searched the
parameter map twice (`find()`, then `operator[]`). That cost 21-43% of
`run()` in earlier commits, and was the top item on this document's "what
would make epiworld faster" list.

epiworld now avoids it. `set_prob_infecting(std::string)` wraps the name in a
`ParamRef` ([`param-ref.hpp`](https://github.com/UofUEpiBio/epiworld/blob/a1ff20a2e347169bb260440c4a9c26918efda7e0/include/epiworld/param-ref.hpp)),
which resolves the name to its position in the model's parameter table on
first use and caches that position, keyed to the model's parameter layout, in
an atomic. Later calls on that model, or on any copy of it (including the
copies `run_multiple()` makes), read the value at that position directly; the
string is looked up again only if the model's parameter layout changed.
epiworldR's `set_prob_infecting_ptr()` and epiworldpy's `set_prob_infecting()`
go through the same call, so every wrapper gets this for free.

Comparing epiworld as benchmarked against the same runner with the
probability set as a plain number now shows a gap within run-to-run noise at
every size: about 8% of `run()` at 10,000 agents, not measurably different at
100,000 (the named-parameter run was fractionally faster than the constant
one there, which is noise, not a real effect), and about 4% at 1,000,000.
Pushing transmission is still a large share of `run()` at 10,000 and 100,000
agents (70% and 61% of the instrumented run below), but that is now the cost
of the transmission computation itself, not of looking up its parameter.

### 2. `run()` re-initializes the population

`Model::run()` starts with `reset()`, which walks the whole population
several times before the first day. It resets each agent, rebuilds the index
of agents by state, recounts the database's totals, clears the queue, and,
to draw the 100 seed cases, builds a list of every agent without a virus:

| Phase of `reset()` | 10,000 agents (ms) | 100,000 agents (ms) | 1,000,000 agents (ms) |
|:---|---:|---:|---:|
| Reset each agent | 0.02 | 0.23 | 3.02 |
| Rebuild the state index | 0.04 | 0.42 | 6.23 |
| Reset the database | 0.02 | 0.15 | 1.70 |
| Clear the queue | 0.02 | 0.15 | 1.56 |
| Seed the initial cases | 0.07 | 0.48 | 4.02 |
| **Total** | **0.17** | **1.44** | **16.53** |

The runs with no transmission show the same from the outside: epiworld's
fixed cost grows with the population (0.31, 1.84, and 17.6 ms), while that of
ixa's `execute()` barely does (0.16, 0.27, and 0.62 ms), because its context
arrives already built.

`reset()` is small next to the epidemic at 10,000 and 100,000 agents. At
1,000,000 agents, with an outbreak of about 6,000 cases, it is well over half
of `run()`. Several of its passes could be merged or skipped: on a
model's first run the agents are already in their initial state, and drawing
100 seed cases does not need a list of a million candidates.

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
  (`next()`) takes well under a microsecond, and applying the day's events
  under a millisecond per run at 100,000 agents. [Scenario 02](../scenario_02/README.md),
  where every engine has to record the same outputs, confirms it: ixa adds
  that bookkeeping at little cost, and the gap does not change.
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
event. The instrumented run spends 4.0 ms placing tools and seeds, against
0.48 ms for the seeds alone in scenario 00: about 3.5 ms for the vaccine. ixa
sets a property on its 30,000 vaccinees in a plan at time 0, which raises the
fixed cost of `execute()` from 0.27 to 1.72 ms, about 1.5 ms for the vaccine.

Median time in milliseconds, scenario 01's model:

| Agents | epiworld `run()` | epiworld, constant probability | ixa `execute()` | ixa, per replicate | epiworld, no transmission | ixa `execute()`, no transmission |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.28 | 2.23 | 1.82 | 4.23 | 0.61 | 0.30 |
| 100,000 | 7.55 | 7.21 | 4.22 | 28.55 | 6.20 | 1.72 |

The heap also matters here. The tool clones are the only large batch of
small allocations inside `run()`, and a fresh C++ process has to grow its
heap to serve them, while R and Python start with a large one. This is why
the C++ runner is slower than epiworldR and epiworldpy in scenarios 01 and
02. Letting glibc keep a large heap in reserve (`MALLOC_TOP_PAD_=268435456`)
brings the C++ runner from 6.99 to 6.34 ms at 100,000 agents, over 15 seeds;
serving large blocks from the heap (`MALLOC_MMAP_THRESHOLD_=1073741824`)
brings it to 5.94 ms. Sharing one tool object among agents would remove these
allocations altogether.

The vaccine shrinks the outbreak, so the fixed cost, with the vaccine,
matters more here: at 100,000 agents it is over 80% of `run()`. ixa's
rebuild still dominates its time per replicate. Scenario 02 runs the same
model and the same epidemics, so the same holds there.

## In each scenario's results

Median simulation time per replicate in the benchmark itself, from each
scenario's report. Like the side measurements, the benchmark runs one
replicate at a time:

| Scenario | Agents | epiworld (C++) | epiworldR | epiworldpy | ixa | ixa / epiworld (C++) |
|:---|---:|---:|---:|---:|---:|---:|
| [00](../scenario_00/README.md) | 10,000 | 4.8 | 5.0 | 4.2 | 7.2 | 1.50 |
| [00](../scenario_00/README.md) | 100,000 | 8.4 | 8.0 | 7.5 | 38.5 | 4.60 |
| [01](../scenario_01/README.md) | 10,000 | 2.4 | 2.0 | 2.1 | 4.3 | 1.76 |
| [01](../scenario_01/README.md) | 100,000 | 7.6 | 7.0 | 6.1 | 33.3 | 4.38 |
| [02](../scenario_02/README.md) | 10,000 | 2.4 | 2.0 | 2.1 | 4.7 | 1.94 |
| [02](../scenario_02/README.md) | 100,000 | 7.4 | 7.0 | 6.1 | 32.7 | 4.45 |
| [03](../scenario_03/README.md) | 1,000,000 | 28.1 | 24.5 | 27.8 | 460.2 | 16.38 |

Times are in milliseconds.

- **epiworld is now faster at every size, including 10,000 agents.** Earlier
  versions of this benchmark, before epiworld resolved named parameters once
  per model instead of on every push-transmission check, showed ixa a little
  faster there. That gap is gone: ixa now takes 1.5 to 1.9 times as long as
  the C++ runner at 10,000 agents.
- **At 100,000 agents epiworld is four to five times faster per replicate.**
  Rebuilding ixa's context now takes several times the whole of epiworld's
  `run()`. The ratio is fairly stable across scenarios 00-02 (4.4 to 4.6),
  since the vaccine shrinks the outbreak but not the rebuild, and both sides
  of the ratio shrink together.
- **At 1,000,000 agents ([scenario 03](../scenario_03/README.md)) the rebuild
  dominates.** ixa takes about 16 times as long as epiworld per replicate,
  although its `execute()` alone is still a little faster than epiworld's
  `run()`, of which `reset()` is now well over half.

## What would make epiworld faster

The first item below is already done, upstream, since the last version of
this analysis; the rest would still be changes to epiworld, not to the
benchmark's runners, which use each engine as its documentation shows:

- ~~Resolve a virus's named parameters once, when the run starts, instead of
  on every call.~~ Done: `ParamRef` now caches a resolved parameter's
  position per model layout (see above).
- Make `reset()` cheaper for large populations: merge its passes over the
  agents, skip resetting agents that are already in their initial state, and
  draw the seed cases without first listing every candidate.
- Share one tool object among the agents that receive it, rather than
  cloning it for each.
