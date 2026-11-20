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
| 3d-1b | 1.0.0+g1b10904 | 00 | 10,000 | 0.00538, 0.00547 | 1.02 | 0.00149, 0.00131 | 0.88 | 8, 8 | 0.98 | 0.385, 0.385 |
| 3d-1b | 1.0.0+g1b10904 | 00 | 100,000 | 0.00925, 0.00985 | 1.06 | 0.0171, 0.0122 | 0.71 | 38, 33 | 0.85 | 0.060, 0.060 |
| 3d-1b | 1.0.0+g1b10904 | 01 | 10,000 | 0.00248, 0.00249 | 1.00 | 0.00133, 0.00112 | 0.85 | 8, 8 | 0.98 | 0.119, 0.119 |
| 3d-1b | 1.0.0+g1b10904 | 01 | 100,000 | 0.00937, 0.00934 | 1.00 | 0.0133, 0.011 | 0.82 | 43, 41 | 0.94 | 0.014, 0.014 |
| 3d-1b | 1.0.0+g1b10904 | 02 | 10,000 | 0.00236, 0.00245 | 1.04 | 0.00129, 0.00111 | 0.86 | 9, 8 | 0.99 | 0.119, 0.119 |
| 3d-1b | 1.0.0+g1b10904 | 02 | 100,000 | 0.009, 0.00921 | 1.02 | 0.0136, 0.0107 | 0.79 | 44, 41 | 0.94 | 0.014, 0.014 |
| 3d-1b | 1.0.0+g1b10904 | 03 | 1,000,000 | 0.0326, 0.0306 | 0.94 | 0.172, 0.137 | 0.80 | 352, 276 | 0.78 | 0.006, 0.006 |
| 3d-1b | 1.0.0+g1b10904 | 04 | 165,865 | 0.185, 0.175 | 0.95 | 0.0166, 0.015 | 0.90 | 90, 88 | 0.97 | 0.622, 0.622 |
| 3d-3a | 1.0.0+g93b9781 | 00 | 10,000 | 0.00476, 0.0048 | 1.01 | 0.00136, 0.00113 | 0.83 | 8, 8 | 0.95 | 0.385, 0.385 |
| 3d-3a | 1.0.0+g93b9781 | 00 | 100,000 | 0.00908, 0.00931 | 1.03 | 0.0153, 0.0106 | 0.69 | 38, 30 | 0.79 | 0.060, 0.060 |
| 3d-3a | 1.0.0+g93b9781 | 01 | 10,000 | 0.00248, 0.00245 | 0.99 | 0.00135, 0.00108 | 0.80 | 8, 8 | 0.94 | 0.119, 0.119 |
| 3d-3a | 1.0.0+g93b9781 | 01 | 100,000 | 0.0111, 0.0115 | 1.04 | 0.0196, 0.0137 | 0.70 | 43, 38 | 0.88 | 0.014, 0.014 |
| 3d-3a | 1.0.0+g93b9781 | 02 | 10,000 | 0.00237, 0.00237 | 1.00 | 0.00128, 0.00102 | 0.80 | 9, 8 | 0.95 | 0.119, 0.119 |
| 3d-3a | 1.0.0+g93b9781 | 02 | 100,000 | 0.0123, 0.0113 | 0.92 | 0.0177, 0.0134 | 0.76 | 44, 38 | 0.88 | 0.014, 0.014 |
| 3d-3a | 1.0.0+g93b9781 | 03 | 1,000,000 | 0.0391, 0.0317 | 0.81 | 0.307, 0.197 | 0.64 | 352, 253 | 0.72 | 0.006, 0.006 |
| 3d-3a | 1.0.0+g93b9781 | 04 | 165,865 | 0.254, 0.232 | 0.91 | 0.0222, 0.0175 | 0.79 | 90, 84 | 0.93 | 0.622, 0.622 |
| 3d-3b | 1.0.0+g07ed7a0 | 00 | 10,000 | 0.00483, 0.00486 | 1.01 | 0.00134, 0.00109 | 0.82 | 8, 8 | 0.95 | 0.385, 0.385 |
| 3d-3b | 1.0.0+g07ed7a0 | 00 | 100,000 | 0.00917, 0.00947 | 1.03 | 0.0139, 0.0107 | 0.77 | 38, 30 | 0.79 | 0.060, 0.060 |
| 3d-3b | 1.0.0+g07ed7a0 | 01 | 10,000 | 0.00242, 0.00243 | 1.00 | 0.00132, 0.00108 | 0.82 | 8, 8 | 0.94 | 0.119, 0.119 |
| 3d-3b | 1.0.0+g07ed7a0 | 01 | 100,000 | 0.00982, 0.00957 | 0.97 | 0.014, 0.0109 | 0.78 | 43, 38 | 0.88 | 0.014, 0.014 |
| 3d-3b | 1.0.0+g07ed7a0 | 02 | 10,000 | 0.00242, 0.00243 | 1.00 | 0.00131, 0.00108 | 0.82 | 9, 8 | 0.95 | 0.119, 0.119 |
| 3d-3b | 1.0.0+g07ed7a0 | 02 | 100,000 | 0.01, 0.00964 | 0.96 | 0.0142, 0.0112 | 0.79 | 44, 38 | 0.88 | 0.014, 0.014 |
| 3d-3b | 1.0.0+g07ed7a0 | 03 | 1,000,000 | 0.033, 0.0301 | 0.91 | 0.167, 0.129 | 0.77 | 352, 253 | 0.72 | 0.006, 0.006 |
| 3d-3b | 1.0.0+g07ed7a0 | 04 | 165,865 | 0.24, 0.202 | 0.84 | 0.0192, 0.0136 | 0.71 | 90, 84 | 0.93 | 0.622, 0.622 |
| nh-2-v2 | 1.0.0+gd571d3e | 00 | 10,000 | 0.00488, 0.00529 | 1.08 | 0.00137, 0.00109 | 0.79 | 8, 7 | 0.81 | 0.385, 0.385 |
| nh-2-v2 | 1.0.0+gd571d3e | 00 | 100,000 | 0.00821, 0.0108 | 1.31 | 0.013, 0.01 | 0.77 | 38, 33 | 0.87 | 0.060, 0.060 |
| nh-2-v2 | 1.0.0+gd571d3e | 01 | 10,000 | 0.00253, 0.00257 | 1.02 | 0.00127, 0.00104 | 0.82 | 8, 7 | 0.91 | 0.119, 0.119 |
| nh-2-v2 | 1.0.0+gd571d3e | 01 | 100,000 | 0.00859, 0.00838 | 0.98 | 0.0123, 0.00977 | 0.80 | 43, 39 | 0.90 | 0.014, 0.014 |
| nh-2-v2 | 1.0.0+gd571d3e | 02 | 10,000 | 0.00236, 0.00258 | 1.09 | 0.00126, 0.00103 | 0.82 | 9, 8 | 0.95 | 0.119, 0.119 |
| nh-2-v2 | 1.0.0+gd571d3e | 02 | 100,000 | 0.00893, 0.00885 | 0.99 | 0.0132, 0.0103 | 0.78 | 44, 40 | 0.91 | 0.014, 0.014 |
| nh-2-v2 | 1.0.0+gd571d3e | 03 | 1,000,000 | 0.0259, 0.0404 | 1.56 | 0.134, 0.112 | 0.84 | 352, 294 | 0.84 | 0.006, 0.006 |
| nh-2-v2 | 1.0.0+gd571d3e | 04 | 165,865 | 0.218, 0.232 | 1.07 | 0.0195, 0.0139 | 0.71 | 90, 52 | 0.57 | 0.622, 0.622 |
| reset-fix | 1.0.0+g384c89f | 00 | 10,000 | 0.00488, 0.00535 | 1.10 | 0.00135, 0.00109 | 0.81 | 8, 7 | 0.80 | 0.385, 0.385 |
| reset-fix | 1.0.0+g384c89f | 00 | 100,000 | 0.00958, 0.0109 | 1.13 | 0.015, 0.0115 | 0.77 | 38, 29 | 0.77 | 0.060, 0.060 |
| reset-fix | 1.0.0+g384c89f | 01 | 10,000 | 0.00245, 0.00263 | 1.07 | 0.00132, 0.00109 | 0.82 | 8, 7 | 0.90 | 0.119, 0.119 |
| reset-fix | 1.0.0+g384c89f | 01 | 100,000 | 0.0103, 0.0097 | 0.94 | 0.0146, 0.0114 | 0.78 | 43, 37 | 0.85 | 0.014, 0.014 |
| reset-fix | 1.0.0+g384c89f | 02 | 10,000 | 0.00244, 0.00264 | 1.08 | 0.0013, 0.0011 | 0.85 | 9, 8 | 0.92 | 0.119, 0.119 |
| reset-fix | 1.0.0+g384c89f | 02 | 100,000 | 0.011, 0.00998 | 0.90 | 0.016, 0.0121 | 0.75 | 44, 37 | 0.86 | 0.014, 0.014 |
| reset-fix | 1.0.0+g384c89f | 03 | 1,000,000 | 0.0292, 0.0274 | 0.94 | 0.159, 0.12 | 0.76 | 352, 253 | 0.72 | 0.006, 0.006 |
| reset-fix | 1.0.0+g384c89f | 04 | 165,865 | 0.249, 0.231 | 0.93 | 0.0195, 0.0138 | 0.71 | 90, 52 | 0.57 | 0.622, 0.622 |
| step-trim | 1.0.0+g3944abd | 00 | 10,000 | 0.00473, 0.00507 | 1.07 | 0.00134, 0.00109 | 0.82 | 8, 7 | 0.80 | 0.385, 0.385 |
| step-trim | 1.0.0+g3944abd | 00 | 100,000 | 0.00958, 0.0108 | 1.13 | 0.0164, 0.012 | 0.73 | 38, 29 | 0.77 | 0.060, 0.060 |
| step-trim | 1.0.0+g3944abd | 01 | 10,000 | 0.00242, 0.00254 | 1.05 | 0.0013, 0.00109 | 0.84 | 8, 7 | 0.90 | 0.119, 0.119 |
| step-trim | 1.0.0+g3944abd | 01 | 100,000 | 0.0112, 0.0101 | 0.90 | 0.0153, 0.0118 | 0.77 | 43, 37 | 0.85 | 0.014, 0.014 |
| step-trim | 1.0.0+g3944abd | 02 | 10,000 | 0.00246, 0.0026 | 1.06 | 0.00133, 0.00111 | 0.84 | 9, 8 | 0.92 | 0.119, 0.119 |
| step-trim | 1.0.0+g3944abd | 02 | 100,000 | 0.011, 0.00995 | 0.90 | 0.0156, 0.0121 | 0.78 | 44, 37 | 0.86 | 0.014, 0.014 |
| step-trim | 1.0.0+g3944abd | 03 | 1,000,000 | 0.0382, 0.0339 | 0.89 | 0.253, 0.159 | 0.63 | 352, 253 | 0.72 | 0.006, 0.006 |
| step-trim | 1.0.0+g3944abd | 04 | 165,865 | 0.269, 0.257 | 0.95 | 0.0223, 0.0176 | 0.79 | 90, 52 | 0.57 | 0.622, 0.622 |
| tracks | 1.0.0+g9e42b2d | 00 | 10,000 | 0.00451, 0.00487 | 1.08 | 0.00119, 0.000996 | 0.84 | 8, 7 | 0.80 | 0.385, 0.385 |
| tracks | 1.0.0+g9e42b2d | 00 | 100,000 | 0.008, 0.00896 | 1.12 | 0.0112, 0.00882 | 0.79 | 38, 29 | 0.77 | 0.060, 0.060 |
| tracks | 1.0.0+g9e42b2d | 01 | 10,000 | 0.0023, 0.00243 | 1.06 | 0.00118, 0.000991 | 0.84 | 8, 7 | 0.90 | 0.119, 0.119 |
| tracks | 1.0.0+g9e42b2d | 01 | 100,000 | 0.00814, 0.00785 | 0.96 | 0.0114, 0.00915 | 0.80 | 43, 37 | 0.85 | 0.014, 0.014 |
| tracks | 1.0.0+g9e42b2d | 02 | 10,000 | 0.00225, 0.00243 | 1.08 | 0.00116, 0.000996 | 0.86 | 8, 8 | 0.92 | 0.119, 0.119 |
| tracks | 1.0.0+g9e42b2d | 02 | 100,000 | 0.00813, 0.00807 | 0.99 | 0.0115, 0.00944 | 0.82 | 43, 37 | 0.86 | 0.014, 0.014 |
| tracks | 1.0.0+g9e42b2d | 03 | 1,000,000 | 0.0279, 0.031 | 1.11 | 0.159, 0.12 | 0.75 | 352, 253 | 0.72 | 0.006, 0.006 |
| tracks | 1.0.0+g9e42b2d | 04 | 165,865 | 0.206, 0.21 | 1.02 | 0.0144, 0.0124 | 0.86 | 90, 52 | 0.57 | 0.622, 0.622 |
| nh-5b-1 | 1.0.0+g883546a | 00 | 10,000 | 0.00434, 0.00476 | 1.10 | 0.00116, 0.000993 | 0.86 | 8, 7 | 0.80 | 0.385, 0.385 |
| nh-5b-1 | 1.0.0+g883546a | 00 | 100,000 | 0.00784, 0.00888 | 1.13 | 0.0112, 0.00921 | 0.82 | 38, 29 | 0.77 | 0.060, 0.060 |
| nh-5b-1 | 1.0.0+g883546a | 01 | 10,000 | 0.00226, 0.00244 | 1.08 | 0.00118, 0.000986 | 0.84 | 8, 7 | 0.90 | 0.119, 0.119 |
| nh-5b-1 | 1.0.0+g883546a | 01 | 100,000 | 0.00746, 0.00748 | 1.00 | 0.011, 0.00887 | 0.81 | 43, 37 | 0.85 | 0.014, 0.014 |
| nh-5b-1 | 1.0.0+g883546a | 02 | 10,000 | 0.00224, 0.00244 | 1.09 | 0.00118, 0.000984 | 0.83 | 8, 8 | 0.92 | 0.119, 0.119 |
| nh-5b-1 | 1.0.0+g883546a | 02 | 100,000 | 0.00739, 0.00763 | 1.03 | 0.011, 0.00891 | 0.81 | 43, 37 | 0.86 | 0.014, 0.014 |
| nh-5b-1 | 1.0.0+g883546a | 03 | 1,000,000 | 0.0281, 0.0281 | 1.00 | 0.135, 0.108 | 0.80 | 352, 253 | 0.72 | 0.006, 0.006 |
| nh-5b-1 | 1.0.0+g883546a | 04 | 165,865 | 0.172, 0.167 | 0.97 | 0.0142, 0.0122 | 0.86 | 90, 52 | 0.57 | 0.622, 0.622 |
| nh-5b-2 | 1.0.0+g83b9110 | 00 | 10,000 | 0.00436, 0.00482 | 1.11 | 0.00114, 0.000958 | 0.84 | 8, 7 | 0.80 | 0.385, 0.385 |
| nh-5b-2 | 1.0.0+g83b9110 | 00 | 100,000 | 0.00801, 0.00896 | 1.12 | 0.0113, 0.00902 | 0.80 | 38, 29 | 0.77 | 0.060, 0.060 |
| nh-5b-2 | 1.0.0+g83b9110 | 01 | 10,000 | 0.00225, 0.00245 | 1.09 | 0.00115, 0.000965 | 0.84 | 8, 7 | 0.91 | 0.119, 0.119 |
| nh-5b-2 | 1.0.0+g83b9110 | 01 | 100,000 | 0.00804, 0.00804 | 1.00 | 0.0114, 0.00911 | 0.80 | 43, 38 | 0.87 | 0.014, 0.014 |
| nh-5b-2 | 1.0.0+g83b9110 | 02 | 10,000 | 0.00223, 0.00244 | 1.09 | 0.00116, 0.000969 | 0.84 | 8, 8 | 0.96 | 0.119, 0.119 |
| nh-5b-2 | 1.0.0+g83b9110 | 02 | 100,000 | 0.00783, 0.00788 | 1.01 | 0.0113, 0.00903 | 0.80 | 43, 38 | 0.88 | 0.014, 0.014 |
| nh-5b-2 | 1.0.0+g83b9110 | 03 | 1,000,000 | 0.0267, 0.0275 | 1.03 | 0.133, 0.108 | 0.81 | 352, 253 | 0.72 | 0.006, 0.006 |
| nh-5b-2 | 1.0.0+g83b9110 | 04 | 165,865 | 0.169, 0.183 | 1.08 | 0.0145, 0.0128 | 0.88 | 90, 52 | 0.57 | 0.622, 0.622 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 00 | 10,000 | 0.00452, 0.00501 | 1.11 | 0.00121, 0.001 | 0.83 | 8, 7 | 0.79 | 0.385, 0.385 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 00 | 100,000 | 0.00809, 0.00922 | 1.14 | 0.011, 0.0089 | 0.81 | 38, 29 | 0.77 | 0.060, 0.060 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 01 | 10,000 | 0.00236, 0.0026 | 1.10 | 0.00123, 0.00103 | 0.83 | 8, 7 | 0.91 | 0.119, 0.119 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 01 | 100,000 | 0.00784, 0.00791 | 1.01 | 0.0111, 0.00896 | 0.81 | 43, 38 | 0.87 | 0.014, 0.014 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 02 | 10,000 | 0.0023, 0.00256 | 1.11 | 0.00122, 0.00101 | 0.83 | 8, 8 | 0.96 | 0.119, 0.119 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 02 | 100,000 | 0.0089, 0.00858 | 0.96 | 0.0126, 0.00994 | 0.79 | 43, 38 | 0.88 | 0.014, 0.014 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 03 | 1,000,000 | 0.0266, 0.0281 | 1.06 | 0.142, 0.108 | 0.76 | 352, 253 | 0.72 | 0.006, 0.006 |
| nh-5b-3 | 1.0.0+gd2df4d9 | 04 | 165,865 | 0.209, 0.2 | 0.96 | 0.0152, 0.0124 | 0.81 | 90, 52 | 0.57 | 0.622, 0.622 |
| 4a | 1.0.0+g6dc373a | 00 | 10,000 | 0.0045, 0.00512 | 1.14 | 0.00126, 0.00106 | 0.84 | 8, 7 | 0.79 | 0.385, 0.385 |
| 4a | 1.0.0+g6dc373a | 00 | 100,000 | 0.00909, 0.00987 | 1.09 | 0.0133, 0.01 | 0.76 | 38, 29 | 0.77 | 0.060, 0.060 |
| 4a | 1.0.0+g6dc373a | 01 | 10,000 | 0.00234, 0.00258 | 1.10 | 0.00128, 0.00106 | 0.83 | 8, 7 | 0.91 | 0.119, 0.119 |
| 4a | 1.0.0+g6dc373a | 01 | 100,000 | 0.00899, 0.00841 | 0.94 | 0.0127, 0.00996 | 0.79 | 43, 38 | 0.87 | 0.014, 0.014 |
| 4a | 1.0.0+g6dc373a | 02 | 10,000 | 0.00234, 0.00262 | 1.12 | 0.00128, 0.00107 | 0.84 | 8, 8 | 0.97 | 0.119, 0.119 |
| 4a | 1.0.0+g6dc373a | 02 | 100,000 | 0.00849, 0.00826 | 0.97 | 0.0125, 0.00983 | 0.79 | 43, 38 | 0.88 | 0.014, 0.014 |
| 4a | 1.0.0+g6dc373a | 03 | 1,000,000 | 0.0284, 0.0271 | 0.95 | 0.162, 0.118 | 0.73 | 352, 253 | 0.72 | 0.006, 0.006 |
| 4a | 1.0.0+g6dc373a | 04 | 165,865 | 0.179, 0.189 | 1.05 | 0.0155, 0.0125 | 0.81 | 90, 52 | 0.57 | 0.622, 0.622 |
| 4b | 1.0.0+ga4cb34d | 00 | 10,000 | 0.00523, 0.00569 | 1.09 | 0.0015, 0.00123 | 0.82 | 8, 7 | 0.80 | 0.385, 0.385 |
| 4b | 1.0.0+ga4cb34d | 00 | 100,000 | 0.00874, 0.0105 | 1.21 | 0.0145, 0.0107 | 0.74 | 38, 29 | 0.77 | 0.060, 0.060 |
| 4b | 1.0.0+ga4cb34d | 01 | 10,000 | 0.00261, 0.00285 | 1.09 | 0.0014, 0.00118 | 0.84 | 8, 7 | 0.91 | 0.119, 0.119 |
| 4b | 1.0.0+ga4cb34d | 01 | 100,000 | 0.00981, 0.0103 | 1.05 | 0.0144, 0.0111 | 0.77 | 43, 38 | 0.89 | 0.014, 0.014 |
| 4b | 1.0.0+ga4cb34d | 02 | 10,000 | 0.00242, 0.00281 | 1.16 | 0.00136, 0.00111 | 0.82 | 8, 8 | 0.97 | 0.119, 0.119 |
| 4b | 1.0.0+ga4cb34d | 02 | 100,000 | 0.0096, 0.00978 | 1.02 | 0.0141, 0.0109 | 0.78 | 43, 39 | 0.90 | 0.014, 0.014 |
| 4b | 1.0.0+ga4cb34d | 03 | 1,000,000 | 0.029, 0.03 | 1.03 | 0.203, 0.132 | 0.65 | 352, 253 | 0.72 | 0.006, 0.006 |
| 4b | 1.0.0+ga4cb34d | 04 | 165,865 | 0.235, 0.261 | 1.11 | 0.0259, 0.0137 | 0.53 | 90, 52 | 0.57 | 0.622, 0.622 |
| 4c | 1.0.0+gfe1e229 | 00 | 10,000 | 0.00491, 0.00547 | 1.11 | 0.00141, 0.00114 | 0.81 | 8, 7 | 0.81 | 0.385, 0.385 |
| 4c | 1.0.0+gfe1e229 | 00 | 100,000 | 0.00849, 0.00989 | 1.17 | 0.014, 0.0104 | 0.74 | 38, 29 | 0.77 | 0.060, 0.060 |
| 4c | 1.0.0+gfe1e229 | 01 | 10,000 | 0.00256, 0.00289 | 1.13 | 0.00138, 0.00118 | 0.86 | 8, 8 | 0.92 | 0.119, 0.119 |
| 4c | 1.0.0+gfe1e229 | 01 | 100,000 | 0.01, 0.0101 | 1.01 | 0.0148, 0.0108 | 0.73 | 43, 38 | 0.89 | 0.014, 0.014 |
| 4c | 1.0.0+gfe1e229 | 02 | 10,000 | 0.00244, 0.00285 | 1.17 | 0.00136, 0.00113 | 0.83 | 8, 8 | 0.96 | 0.119, 0.119 |
| 4c | 1.0.0+gfe1e229 | 02 | 100,000 | 0.00926, 0.00923 | 1.00 | 0.0133, 0.0103 | 0.77 | 43, 39 | 0.90 | 0.014, 0.014 |
| 4c | 1.0.0+gfe1e229 | 03 | 1,000,000 | 0.0344, 0.0318 | 0.92 | 0.192, 0.147 | 0.77 | 352, 253 | 0.72 | 0.006, 0.006 |
| 4c | 1.0.0+gfe1e229 | 04 | 165,865 | 0.212, 0.223 | 1.06 | 0.0189, 0.0158 | 0.83 | 90, 52 | 0.57 | 0.622, 0.622 |
| 4d | 1.0.0+g493382b | 00 | 10,000 | 0.00497, 0.00551 | 1.11 | 0.0014, 0.00111 | 0.80 | 8, 7 | 0.79 | 0.385, 0.385 |
| 4d | 1.0.0+g493382b | 00 | 100,000 | 0.00915, 0.0109 | 1.19 | 0.017, 0.0115 | 0.68 | 38, 29 | 0.77 | 0.060, 0.060 |
| 4d | 1.0.0+g493382b | 01 | 10,000 | 0.00253, 0.00294 | 1.16 | 0.00135, 0.00113 | 0.84 | 8, 8 | 0.92 | 0.119, 0.119 |
| 4d | 1.0.0+g493382b | 01 | 100,000 | 0.0105, 0.0102 | 0.97 | 0.0162, 0.0114 | 0.70 | 43, 38 | 0.89 | 0.014, 0.014 |
| 4d | 1.0.0+g493382b | 02 | 10,000 | 0.00242, 0.00283 | 1.17 | 0.00132, 0.0011 | 0.84 | 8, 8 | 0.96 | 0.119, 0.119 |
| 4d | 1.0.0+g493382b | 02 | 100,000 | 0.0117, 0.011 | 0.94 | 0.0164, 0.0119 | 0.73 | 43, 39 | 0.90 | 0.014, 0.014 |
| 4d | 1.0.0+g493382b | 03 | 1,000,000 | 0.0305, 0.0298 | 0.98 | 0.166, 0.133 | 0.80 | 352, 253 | 0.72 | 0.006, 0.006 |
| 4d | 1.0.0+g493382b | 04 | 165,865 | 0.292, 0.318 | 1.09 | 0.0242, 0.0177 | 0.73 | 90, 52 | 0.57 | 0.622, 0.622 |
| 5 | 1.0.0+gda6bdcb | 00 | 10,000 | 0.00453, 0.00558 | 1.23 | 0.00126, 0.00104 | 0.83 | 8, 7 | 0.79 | 0.385, 0.385 |
| 5 | 1.0.0+gda6bdcb | 00 | 100,000 | 0.00848, 0.0104 | 1.23 | 0.0141, 0.0107 | 0.76 | 38, 29 | 0.77 | 0.060, 0.060 |
| 5 | 1.0.0+gda6bdcb | 01 | 10,000 | 0.00248, 0.00292 | 1.18 | 0.00132, 0.00109 | 0.83 | 8, 7 | 0.91 | 0.119, 0.119 |
| 5 | 1.0.0+gda6bdcb | 01 | 100,000 | 0.00909, 0.00923 | 1.02 | 0.0136, 0.0101 | 0.74 | 43, 38 | 0.89 | 0.014, 0.014 |
| 5 | 1.0.0+gda6bdcb | 02 | 10,000 | 0.00239, 0.0029 | 1.22 | 0.00128, 0.00106 | 0.83 | 8, 8 | 0.96 | 0.119, 0.119 |
| 5 | 1.0.0+gda6bdcb | 02 | 100,000 | 0.00917, 0.00947 | 1.03 | 0.0132, 0.0103 | 0.78 | 43, 39 | 0.90 | 0.014, 0.014 |
| 5 | 1.0.0+gda6bdcb | 03 | 1,000,000 | 0.0323, 0.0316 | 0.98 | 0.208, 0.175 | 0.84 | 352, 253 | 0.72 | 0.006, 0.006 |
| 5 | 1.0.0+gda6bdcb | 04 | 165,865 | 0.219, 0.231 | 1.06 | 0.0212, 0.0141 | 0.66 | 90, 52 | 0.57 | 0.622, 0.622 |
| 6a | 1.0.0+gbb78cbb | 00 | 10,000 | 0.00465, 0.00602 | 1.30 | 0.00122, 0.00105 | 0.86 | 8, 7 | 0.80 | 0.385, 0.397 |
| 6a | 1.0.0+gbb78cbb | 00 | 100,000 | 0.00891, 0.0117 | 1.31 | 0.013, 0.0101 | 0.77 | 38, 29 | 0.77 | 0.060, 0.060 |
| 6a | 1.0.0+gbb78cbb | 01 | 10,000 | 0.00254, 0.00304 | 1.19 | 0.00125, 0.00106 | 0.85 | 8, 7 | 0.91 | 0.119, 0.119 |
| 6a | 1.0.0+gbb78cbb | 01 | 100,000 | 0.00896, 0.00956 | 1.07 | 0.013, 0.00995 | 0.76 | 43, 38 | 0.89 | 0.014, 0.014 |
| 6a | 1.0.0+gbb78cbb | 02 | 10,000 | 0.00232, 0.00285 | 1.23 | 0.00118, 0.00099 | 0.84 | 8, 8 | 0.96 | 0.119, 0.119 |
| 6a | 1.0.0+gbb78cbb | 02 | 100,000 | 0.00927, 0.00983 | 1.06 | 0.013, 0.0104 | 0.80 | 43, 39 | 0.90 | 0.014, 0.014 |
| 6a | 1.0.0+gbb78cbb | 03 | 1,000,000 | 0.0289, 0.0313 | 1.08 | 0.154, 0.112 | 0.73 | 352, 253 | 0.72 | 0.006, 0.007 |
| 6a | 1.0.0+gbb78cbb | 04 | 165,865 | 0.235, 0.301 | 1.28 | 0.0202, 0.0141 | 0.70 | 90, 52 | 0.57 | 0.622, 0.623 |
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

### 3d-1b

epiworld `v1-layers` at
[`1b10904`](https://github.com/UofUEpiBio/epiworld/commit/1b10904d8fb11e4d775a9eca154305f0b6cd9e22),
which adds Phase 3d-2 (UofUEpiBio/epiworld#340, #345: the measles models,
which these scenarios do not use) and 3d-1b (UofUEpiBio/epiworld#341: networks
built without `std::map`, the fix for the 3d-1 finding,
UofUEpiBio/epiworld#339) to the previous stage. Against epiworld 0.17.1.

- **Results agree.** The median attack rates are the same in every scenario
  and size, and the same as at 3d-1.
- **Building the network is back below 0.x:** the build ratio fell from 5.2-9.0
  to 0.71-0.90, and peak memory from 1.06-2.56 times 0.x's to 0.78-0.99. At
  1,000,000 agents the build takes 0.137 s (0.x: 0.172) and the setup peaks at
  276 MiB (0.x: 352; 3d-1: 901).
- **1,000,000 agents is no longer noisy:** the simulation ratio is 0.94
  (median 0.031 s), against 1.58 on scattered replicates at 3d-1.
- **The simulation ratio grew by more than 10% up to 165,865 agents** (to
  1.00-1.06, from 0.69-0.97; 0.95 on the GeoPops network, from 0.81), so it
  was profiled before the next phase. The time per step did not change; page
  faults moved from the build into the simulation. The 3d-1 build freed tens
  of MiB of map nodes that glibc kept, and the run's first allocations (the
  reset inside `run()`, the database, the queue) reused them. Built against
  the three commits, scenario_01's runner at 100,000 agents (40 interleaved
  replicates in this image) takes 0.0069 s to simulate at 3d-1 (`5811a28`),
  0.0070 s after 3d-2 (`b699adb`), and 0.0101 s now. It takes 0 minor
  page faults inside `run()` before #341 and about 3,400 (13 MiB) after; at
  under a microsecond each, that is the difference. So the 3d-1 ratios below 1
  were the same artifact, and the two engines now simulate at parity. Build
  plus simulation at 100,000 agents takes 0.020 s against 0.104 s at 3d-1.
  Nothing to fix in epiworld.

### 3d-3a

epiworld `v1-layers` at
[`93b9781d`](https://github.com/UofUEpiBio/epiworld/commit/93b9781df42e4e51818cf960da20274f359452cd)
(Phase 3d-3a, UofUEpiBio/epiworld#349, plus the `master` merge #351),
against epiworld 0.17.1.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No ratio grew by more than 10% from 3d-1b.** Simulation ratios range from
  0.81 to 1.04 (previously 0.94 to 1.06), and peak-memory ratios all fell.
  The 1,000,000-agent simulation ratio is 0.81 (0.032 s versus 0.039 s), so
  the row remains stable rather than reproducing the earlier noise.

### NH-2, V-1, V-2

epiworld `v1-layers` at
[`d571d3e9`](https://github.com/UofUEpiBio/epiworld/commit/d571d3e978b54d0e3a8e7a82147dcf39103ba124)
(natural histories, UofUEpiBio/epiworld#363; variants by id, #364; no `TSeq`
template, #365), against epiworld 0.17.1. The runners declare the SEIRH as a
natural history (the same rates, the two out of Infected competing as before)
and use the plain classes.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **The simulation ratio grew by more than 10% at 100,000 agents in
  scenario_00 (1.03 to 1.31), at 1,000,000 (0.91 to 1.56) and on the GeoPops
  network (0.84 to 1.07)**, so it was profiled. Build and peak-memory ratios
  did not grow (the GeoPops peak fell from 84 to 52 MiB).
  - Built against the three merges, scenario_03's runner (10 interleaved
    replicates in this image) simulates in 0.0378 s after #363, 0.0383 s after
    #364 and 0.0381 s after #365, with identical epidemics: V-1 and V-2 change
    nothing. Against 3d-3b (`07ed7a0`, the previous runner) it simulates in
    0.0253 s, and the current one in 0.0375 s, interleaved.
  - Callgrind on `run()` (one replicate): 444M instructions at 3d-3b, 526M
    now, of which `reset()` (inside `run()`, so inside the simulation time) is
    79M and 153M. The steps changed little (+8M). The reset now makes several
    passes over the population and allocates new vectors of its size at every
    run: the eligibility of each agent for the virus (a stage lookup per
    agent), a label per agent and a pool of the eligible ones
    (`assign::detail::draw_counts()`), and the natural history's per-agent
    records; the extra allocations (about 30M instructions in `operator new`,
    and their page faults, which callgrind does not count) explain why the
    time grew more than the instructions.
  - To fix in epiworld before the next C++ phase: a placement of a few agents
    should not cost passes and fresh allocations over the whole population at
    every reset.

### Reset fix

epiworld `v1-layers` at
[`384c89fe`](https://github.com/UofUEpiBio/epiworld/commit/384c89feb46484177f4a1ea88a70de7de380aadf)
(the reset without population-sized allocations, UofUEpiBio/epiworld#367,
after the docs of #366), against epiworld 0.17.1. The runners are unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No ratio grew by more than 10% from the previous stage.** At 1,000,000
  agents the simulation ratio is back from 1.56 to 0.94 (3d-3b: 0.91): the
  natural history's per-agent records had made each run first-touch about
  40 MB (page faults during `run()`: 9,350 at 3d-3b, 20,348 at NH-2/V-2, 10,536
  now, in this image).
- **At 100,000 agents in scenario_00 (1.13) and on the GeoPops network (0.93)
  the ratios fell, but only because this run's 0.x times were slower**: the
  1.0 times did not change from the previous stage (0.0108 to 0.0109 s; 0.232 to
  0.231 s), and stay about 15% above 3d-3b (0.00947 s; 0.202 s).
  - scenario_00's runner at 100,000 agents (20 interleaved replicates in this
    image): 0.00922 s at 3d-3b (`07ed7a0`), 0.01026 s now, with identical
    epidemics and fewer page faults (1,373 and 1,141).
  - Callgrind on `run()` (one replicate): 121.5M instructions at 3d-3b, 130.0M
    now; `reset()` is 8.1M and 9.4M. The rest is the natural history's step:
    each day's proposal (`nh_step()`, `nh_propose()`, the transitions' values,
    `Value::eval()`), which grows with the agents that move (GeoPops' attack
    rate is 62%).

### Step trim

epiworld `v1-layers` at
[`3944abd`](https://github.com/UofUEpiBio/epiworld/commit/3944abd3b322392b002c47f4949383695593e417)
(the natural history's daily step trimmed, UofUEpiBio/epiworld#368), against
epiworld 0.17.1. The runners are unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No ratio grew by more than 10% from the previous stage.** The largest
  change is the GeoPops network's, 0.93 to 0.95.
- **This run cannot see #368's gain.** It removed about 4% of `run()`'s
  instructions at 100,000 agents in scenario_00 (130.0M to 124.6M; 3d-3b:
  121.5M), within the noise of a stage. At that size the 1.0 time did not move
  (0.0109 to 0.0108 s) and the ratio stays 1.13 (3d-3b: 1.03). On GeoPops both
  engines were slower in this run (0.x 0.249 to 0.269 s, 1.0 0.231 to 0.257 s).
  - Runners compared directly (16 interleaved seeds in this image): 0.00893 s
    at 3d-3b and 0.00926 s at #368 for scenario_00 at 100,000 agents; 0.156 s
    and 0.179 s on GeoPops. The epidemics are identical.
- **The remaining cost is per infected agent, not per contact or layer.**
  Callgrind on `run()` (one seed; 32 KiB L1 and 8 MiB last-level caches),
  #368 against 3d-3b:
  - GeoPops: +3.5% instructions, +17% last-level data misses.
  - A synthetic population of the same size with twice GeoPops' mean degree
    (9.2: households, random groups of about 30 and local community contacts;
    73% infected by day 100): +2.2% instructions and +4% data misses, both as
    one network and as three network layers.
  - The extra work is about the same in both (+34M and +32M instructions, about
    300 per infection), so denser networks dilute it. GeoPops, sparse with a
    62% attack rate, is the worst case.
  - Most of GeoPops' extra misses come from reading each infected agent's own
    copy of its virus during the daily step. Reserving the natural history's
    records at reset changed nothing.

### Tracks

epiworld `v1-layers` at
[`9e42b2d`](https://github.com/UofUEpiBio/epiworld/commit/9e42b2dce88ce4fdd9c82bba285d77a0a0c58c25)
(NH-5a, the engine's per-disease state in a `Track`, UofUEpiBio/epiworld#372,
and the database in parts, #373; both refactors with identical results),
against epiworld 0.17.1. The runners are unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **One ratio grew by more than 10%, and it is the 0.x run's doing.**
  scenario_03 (1,000,000 agents) went from 0.89 to 1.11, and scenario_02 at
  100,000 agents from 0.90 to 0.99. In every row the 1.0 engine was faster
  than in the previous stage (scenario_03: 0.0339 to 0.0310 s), but 0.x was
  faster still (0.0382 to 0.0279 s, 27%), with the same 0.x code; the
  environment's fingerprint changed too.
- **The cost of the two refactors, counted:** callgrind on the same SEIR
  model on a small-world network (degree 10, 100 days), #368's commit against
  this one, with the benchmark's flags: +1.7% instructions at 100,000 agents
  (1.179 to 1.199 billion) and +1.8% at 1,000,000 (11.84 to 12.05 billion),
  with identical epidemics. That is the cost of looping over the tracks in the
  daily step, and it does not grow with the population.

### NH-5b-1

epiworld `v1-layers` at
[`883546a`](https://github.com/UofUEpiBio/epiworld/commit/883546adb4e8ed8fc485cbf45d3ff02f4e7cfed1)
(NH-5b-1, the outputs by disease and the transition matrices by margin,
UofUEpiBio/epiworld#375, identical numbers; and #376, a standard constant
instead of `M_PI`), against epiworld 0.17.1. The runners are unchanged: the
database functions they call only gained defaulted arguments.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No ratio grew by more than 10%.** The largest change is scenario_01 at
  100,000 agents, 0.96 to 1.00; scenario_03 went back from 1.11 to 1.00 (the
  0.x run's time is back to the earlier stages').
- **The cost, counted:** callgrind with the benchmark's flags, #373's commit
  against this one: the SEIR model on a small-world network (degree 10, 100
  days, 100,000 agents) takes 1,123,456,122 and 1,123,456,167 instructions
  (+45), and the measles mixing model 10,848,004,469 and 10,848,004,597.
- A first run of this stage shared the machine with other work (every row,
  0.x's too, about 2.5 times slower); it was discarded and rerun alone.

### NH-5b-2

epiworld `v1-layers` at
[`83b9110`](https://github.com/UofUEpiBio/epiworld/commit/83b91100ec8428a5105bd8ad8541d50f42d821a2)
(NH-5b-2, several diseases in the engine, UofUEpiBio/epiworld#377; identical
results with one disease), against epiworld 0.17.1. The runners are
unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **One ratio grew by more than 10%: scenario_04 (GeoPops, 165,865 agents),
  0.97 to 1.08** (1.0's simulation time 0.167 to 0.183 s; 0.x's 0.172 to
  0.169). The other rows moved by at most 0.03.
- **The cost, counted:** callgrind of scenario_04's runner (benchmark flags,
  the first replicate's seed), #376's commit against this one: 1,922,890,089
  and 1,978,285,429 instructions for the whole process (+2.9%; the epidemics
  are identical). The difference is in the simulation: the one-disease stage
  update, which the compiler inlined into the state's update function, is now
  a call to the general `nh_step_stage()` (about +40 million), and
  `nh_react()` and the database update are calls too (about +15 million).
  The time grew more than the instructions (about +4.5% of the
  simulation's), as at NH-2 on this network.
- **Next:** an epiworld PR that gives one disease its own inlinable stage step,
  then a rerun of this stage.
- A first run of this stage shared the machine with other work (examples
  compiled and run in another container); it was discarded and rerun alone.
  Its scenario_04 ratio was 1.10.

### NH-5b-3

epiworld `v1-layers` at
[`d2df4d9`](https://github.com/UofUEpiBio/epiworld/commit/d2df4d9302ddb5c169918a3ddeeb27f0b2b072e3):
NH-5b-3 (transitions by strain, UofUEpiBio/epiworld#378) and the fix of
NH-5b-2's regression (an inlinable stage step for one disease,
UofUEpiBio/epiworld#379), against epiworld 0.17.1. The runners are
unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No ratio grew by more than 10%.** The largest growths are scenario_03
  (1,000,000 agents), 1.03 to 1.06, and scenario_00 at 100,000, 1.12 to 1.14.
- **Scenario_04 (GeoPops, 165,865 agents) went back down, 1.08 to 0.96.**
  This is more than #379's own measurement predicted (about 1.01; it measured
  scenario_04's simulation at +4.3% against #376's commit). Both engines were
  slower in this run (0.x's median 0.169 to 0.209 s, 1.0's 0.183 to 0.200 s),
  and the replicates interleave the two engines, so the ratio is the
  measure; read it as "within the rule", not as 1.0 being faster here.
- Scenario_00 has been at 1.10–1.14 since NH-2 (`nh-2-v2`); no stage since
  moved it by more than 3%.

### 4a

epiworld `v1-layers` at
[`6dc373a`](https://github.com/UofUEpiBio/epiworld/commit/6dc373af646fe3cbdcd871b43037f8992d6e84c3)
(Phase 4a, participation: a layer excludes statuses, closing a layer,
UofUEpiBio/epiworld#380; identical results for models that use neither),
against epiworld 0.17.1. The runners are unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **One ratio grew by just over 10%: scenario_04 (GeoPops, 165,865 agents),
  0.96 to 1.05** (1.100 times). It was profiled, as the rule asks:
  - callgrind of scenario_04's runner (benchmark flags, the first
    replicate's seed), #379's merge against #380's: 1,960,098,919 and
    1,992,677,213 instructions for the whole process (+1.7%; identical
    epidemics). The push loop grew, and the compiler stopped inlining
    `transmission_reduction_mixer()` into it, plus a redundant exclusion check
    per carrier and layer. The 4b PR uses the layer's own mask there:
    1,987,736,912 (+1.4%).
  - **Timed, interleaved** (30 replicates, the three builds in turn, the same
    seeds): the simulation's median is 0.2435 s before #380, 0.2362 s at #380
    and 0.2419 s with 4b's change; the median of the paired ratios against
    before #380 is 0.99 and 1.01. No slowdown: the stage's jump is run-to-run
    noise on top of the previous stage's low 0.96 (NH-5b-2 had 1.08).
- The other rows moved within their usual noise (largest: scenario_03,
  1,000,000 agents, 1.06 to 0.96; scenario_01 at 100,000, 1.01 to 0.94).

### 4b

epiworld `v1-layers` at
[`a4cb34d`](https://github.com/UofUEpiBio/epiworld/commit/a4cb34dc28b3e681c1b13002a8378a8d8ebadd44)
(Phase 4b, tools that target layers, UofUEpiBio/epiworld#381; identical
results for every registered model), against epiworld 0.17.1. The runners are
unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **Two ratios grew by more than 10%: scenario_00 at 100,000 agents, 1.09 to
  1.21, and scenario_01 at 100,000, 0.94 to 1.05.** They were profiled, as the
  rule asks: callgrind of each runner (benchmark flags at 100,000 agents, the
  first replicate's seed), #380's merge against #381's, gives 1,216,316,319
  and 1,217,754,473 instructions for scenario_00 (+0.12%) and 1,196,233,047
  and 1,198,277,601 for scenario_01 (+0.17%), with identical epidemics. 4b
  adds a layer check per tool in the contact mixers; scenario_00 has no tools,
  so it does not run there. No slowdown: scenario_00 sat at 1.12 to 1.14 over
  the previous eight stages (4a's 1.09 was low), and scenario_01 at 0.90 to
  1.04.
- The other rows moved within their usual noise (scenario_04, 1.05 to 1.11;
  scenario_03, 0.95 to 1.03).

### 4c

epiworld `v1-layers` at
[`fe1e229`](https://github.com/UofUEpiBio/epiworld/commit/fe1e229d7a653945a126b195a6014344e9010a9d)
(Phase 4c, schedules, persistence and parameter cycles,
UofUEpiBio/epiworld#383; identical results for every registered model),
against epiworld 0.17.1. The runners are unchanged.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No simulate ratio grew by more than 10%.** They moved within -11% and
  +3% (scenario_03 at 1,000,000 agents, 1.03 to 0.92).
- **Two build ratios grew by more than 10%: scenario_04, 0.53 to 0.83, and
  scenario_03, 0.65 to 0.77.** Most of scenario_04's change is the 0.x
  engine's build time falling (0.0259 s to 0.0189 s, with no code change);
  v1's own build times rose by 11-15% (2-15 ms). Profiled: callgrind of each
  runner (benchmark flags, the first replicate's seed), #381's merge against
  #383's, gives 11,865,920,197 and 11,866,898,000 instructions for
  scenario_03 (+0.008%) and 1,244,365,383 and 1,244,572,297 for scenario_04
  (+0.017%), whole runs included. 4c adds, per run, an empty loop over the
  parameter cycles each day and a check for persistence at reset; noise.

### 4d

epiworld `v1-layers` at
[`493382b`](https://github.com/UofUEpiBio/epiworld/commit/493382b97f299d283386888b649a0eb37b6a2d76)
(Phase 4d: the STERGM rewirer example, and `Model::run()` clearing its
running flag with a `try`/`catch`, UofUEpiBio/epiworld#384; identical
results), against epiworld 0.17.1. The runners are unchanged; the reports
are re-rendered.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **No ratio grew by more than 10%** from 4c (simulate: -6% to +7%; build:
  -12% to +4%).
- **Against 3d-3b** (the last stage at parity with 0.x), the small scenarios
  stay 10-17% slower (scenario_00 at 10,000 and 100,000, 01 and 02 at
  10,000), as at 4c; instruction counts put the gap at a few percent (see
  4c). The others are within noise of 3d-3b.

### 5

epiworld `v1-layers` at
[`da6bdcb`](https://github.com/UofUEpiBio/epiworld/commit/da6bdcb67ee15dc6d9b9f6973d51c29499089aca)
(Phase 5: models with a mixing layer can push, UofUEpiBio/epiworld#386;
identical results for network-only models), against epiworld 0.17.1. The
runners are unchanged; the reports are re-rendered.

- **Results agree.** The median attack rates are the same in every scenario
  and size.
- **Two simulate ratios grew by more than 10% from 4d: scenario_00 at 10,000
  (1.11 to 1.23) and scenario_02 at 100,000 (0.94 to 1.03).** The others
  moved within -3% and +5%. Build ratios: -10% to +12%, each within a few
  milliseconds.
- **Profiled, and it is real, if small.** Callgrind of each runner
  (benchmark flags, the first replicate's seed), 4d against 5: scenario_00
  at 10,000 goes from 162,915,496 to 173,010,497 instructions (+6.2%), and
  scenario_01 at 10,000 +2.4%. At 100,000 the gaps are +1.4% and +0.2%.
  Final sizes are identical. The push loop now handles each reached agent in
  a lambda that both kinds of layer share, and the compiler no longer
  inlines it, or the tool reductions it calls. The fix (one loop over a span
  of reached agents) goes in with Phase 6.
- **Against 3d-3b** (the last stage at parity with 0.x), the small scenarios
  are 15-21% slower (scenario_00 at 10,000 and 100,000, 01 and 02 at
  10,000). The others are within noise of 3d-3b (-4% to +14%).

### 6a

epiworld `v1-layers` at
[`bb78cbb`](https://github.com/UofUEpiBio/epiworld/commit/bb78cbbd4398ddf78a48831e678d40b1d5aa25f4)
(Phase 6a: exposures are independent, UofUEpiBio/epiworld#388), against
epiworld 0.17.1. The runners are unchanged; the reports are re-rendered.

- **Results change, as planned, and stay within the other engines'.** A
  susceptible agent now escapes infection with probability `prod(1 - p)`;
  0.x conditioned the draw on at most one transmission. The median attack
  rates move up where agents have several infectious contacts a day. The
  largest move is scenario_00 at 10,000 (0.385 to 0.397). scenario_03 goes
  from 0.0062 to 0.0067; the rest move by 0.001 or less. Each stays within
  the range of the other engines' medians except scenario_00 at 10,000,
  which is just above it: 0.39695 against epydemic's 0.39575 (uncalibrated),
  with the others between 0.378 and 0.392.
- **No calibration factor.** The protocol gives a factor to an engine that
  leaves the agreement the others meet with epiworldR. epiworld-v1 is about
  as far from epiworldR as epydemic, which needs none (scenario_00 at
  10,000: +0.012 and +0.010).
- **Two simulate ratios grew by more than 10% from 5: scenario_04, 1.06 to
  1.28, and scenario_03, 0.98 to 1.08.** The others moved within +1% and
  +7%.
- **Profiled: `log1p` per contact.** Callgrind of scenario_04 (the first
  replicate's seed), Phase 5 with #388's loop fix against 6a:
  - 1,247,920,467 to 1,334,118,010 instructions;
  - `log1p` alone is 34,000,461 of them;
  - the rest grow with the outbreak, which has 6.8% more recoveries in this
    replicate (15,472 to 16,525).

  Contacts usually share their probability, so caching the last hazard
  gives identical results with one `log1p` per distinct probability. With
  the cache, the same run takes 1,294,927,791 instructions and ends with the
  same 16,525 recoveries. The fix goes in with Phase 6b
  (UofUEpiBio/epiworld, branch `phase-6b`), and the next rerun will show it.
- **Against 3d-3b**, the last stage at parity, the simulate ratios are now
  between +9% and +52%. Part of that is the larger outbreaks, which
  independent exposures cause by design.
