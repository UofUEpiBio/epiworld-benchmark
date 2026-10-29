# epiworld 1.0.0: tracking the refactor

[Back to the project overview](../README.md)

epiworld 1.0.0 replaces the network inside each agent with contact layers
(several networks and mixing structures combined in one model). The work
happens on epiworld's
[`v1-layers`](https://github.com/UofUEpiBio/epiworld/tree/v1-layers) branch,
one phase per pull request. Its plan is `contact-layers.md` on that branch:
Section 8 says when this benchmark is rerun, and Section 11.7 says how.

On this repository's `v1-layers` branch, the benchmark has one more engine,
**`epiworld-v1`**. Every scenario has a runner for it,
`scenario_NN/runners/epiworld-v1/main.cpp`. The runner builds the same model
as the `epiworld` runner next to it, through the 1.0.0 API, and takes the same
flags and writes the same record. The two runners differ in three places:

- the population is set in the constructor, `Model<> model(n)`;
- the network is a contact layer,
  `model.add_layer(layer::network<>("contacts", AdjList(...)))`, in place of
  `agents_from_edgelist()`;
- only infected agents transmit, through `set_infectious_states({2})` with the
  default susceptible update, in place of
  `sampler::make_update_susceptible({1, 3, 4})`, which 1.0.0 no longer has.
  Both forms let epiworld choose between pushing and pulling transmission.

The other engines, `epiworld` (0.x) among them, keep their pins. epiworldR and
epiworldpy join at the release; until then only the C++ library is tracked.

## A stage

Each merged phase on epiworld's `v1-layers` is a stage. The pin of
`epiworld-v1` moves to the phase's merge commit (`EPIWORLD_V1_REF` in the
[Makefile](../Makefile)), and then, with nothing else running:

```sh
make container-image
STAGE=3d-1 make container-v1-stage
```

This reruns both C++ engines, `epiworld` and `epiworld-v1`, at the full design
of every scenario, in one invocation. The 0.x engine is rerun too (`--force`),
so the two always share a machine and its load. Replicates run one at a time,
with the engines' replicates interleaved. Then
`scripts/epiworld_v1_history.py` writes the medians per scenario and size to
[`results/epiworld-v1-history.csv`](../results/epiworld-v1-history.csv) under
the stage's name, and rewrites the table below. The run's environment
record (machine, toolchains, pins) is kept as
`results/epiworld-v1-environments/<stage>.json`; the records in
`results/environments/` keep describing the published runs of every engine.

The ratio of the 1.0 median to the 0.x median is what is compared from one
stage to the next. Absolute times depend on the machine. The ratio's level
also depends on the 0.x pin, which includes fewer of epiworld's `master`
changes than `v1-layers` does.

What counts, from the plan:

- until the phase that changes how exposures combine (Phase 6), the two
  engines' attack rates agree within noise;
- a simulation-time or peak-memory ratio that grows by more than 10% from the
  previous stage, in any scenario and size, is profiled before the next phase
  merges, and the finding goes in the stage's note below.

## History

Medians over replicates. **Build** is setup without reading the edge list:
building the model and its network. **Peak** is the overall peak resident
memory of the process. The [methods](methods.md#measures) define each
measure.

<!-- history:begin -->
| Stage | epiworld-v1 | Scenario | Agents | Simulate (s): 0.x, 1.0 | Ratio | Build (s): 0.x, 1.0 | Ratio | Peak (MiB): 0.x, 1.0 | Ratio | Attack rate: 0.x, 1.0 |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 3d-1 | 1.0.0+g5811a28 | 00 | 10,000 | 0.0051, 0.0046 | 0.90 | 0.00137, 0.00758 | 5.55 | 8, 12 | 1.48 | 0.385, 0.385 |
| 3d-1 | 1.0.0+g5811a28 | 00 | 100,000 | 0.00967, 0.00938 | 0.97 | 0.0179, 0.105 | 5.87 | 38, 93 | 2.44 | 0.060, 0.060 |
| 3d-1 | 1.0.0+g5811a28 | 01 | 10,000 | 0.00273, 0.00231 | 0.85 | 0.00146, 0.00767 | 5.24 | 8, 12 | 1.51 | 0.119, 0.119 |
| 3d-1 | 1.0.0+g5811a28 | 01 | 100,000 | 0.0101, 0.00726 | 0.72 | 0.0145, 0.0964 | 6.63 | 43, 93 | 2.16 | 0.014, 0.014 |
| 3d-1 | 1.0.0+g5811a28 | 02 | 10,000 | 0.00266, 0.00227 | 0.85 | 0.00144, 0.00764 | 5.30 | 8, 12 | 1.48 | 0.119, 0.119 |
| 3d-1 | 1.0.0+g5811a28 | 02 | 100,000 | 0.0109, 0.00749 | 0.69 | 0.0159, 0.101 | 6.38 | 43, 93 | 2.15 | 0.014, 0.014 |
| 3d-1 | 1.0.0+g5811a28 | 03 | 1,000,000 | 0.0318, 0.0502 | 1.58 | 0.243, 1.36 | 5.59 | 352, 901 | 2.56 | 0.006, 0.006 |
| 3d-1 | 1.0.0+g5811a28 | 04 | 165,865 | 0.274, 0.221 | 0.81 | 0.0309, 0.278 | 8.97 | 90, 96 | 1.06 | 0.622, 0.622 |
<!-- history:end -->

## Notes by stage

### 3d-1 (catch-up)

epiworld `v1-layers` at
[`5811a28`](https://github.com/UofUEpiBio/epiworld/commit/5811a28aefe233b492976b21f038229fc212096e)
(Phase 3d-1, UofUEpiBio/epiworld#335 and #336), against epiworld 0.17.1. The
first stage, so there is no previous ratio to compare with.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **Simulation is as fast or faster**, by 3% to 31% up to 165,865 agents. At
  1,000,000 agents the median is 1.58 times 0.x's, but on 20 replicates
  whose 1.0 times range from 0.029 to 0.28 s (0.x: 0.027 to 0.05 s); this
  size is rechecked at the next stage before it is read as a regression.
- **Building the network costs more: 5 to 9 times the build time, and up to
  2.6 times the peak memory** (1.06 on the GeoPops network). The runner builds an `AdjList` (a `std::map`
  per agent) and the layer copies it into its ties. At 1,000,000 agents the
  setup peaks at 901 MiB (0.x: 352). After setup the process still holds 779
  MiB (0.x: 275); with `malloc_trim(0)` after building the layer it holds
  314 MiB. So the layer itself costs about 40 MiB more than 0.x, and the rest
  is the `AdjList`'s freed memory, which glibc keeps resident. A
  `layer::network()` that takes the edge list directly (as 0.x's
  `agents_from_edgelist()` did), without the `AdjList`, would remove both the
  build time and the peak; that is for epiworld to decide.
