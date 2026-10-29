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
<!-- history:end -->

## Notes by stage
