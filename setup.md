# Network epidemic ABM speed benchmark

This folder contains a reproducible, resumable benchmark of nine epidemic
simulation engines across several scenarios of increasing complexity:

- **epiworld 0.17.0** (C++), the header-only library, using a custom
  discrete-time SEIRH model built from its state update functions;
- **epiworldR 0.17.0.0** (R wrapper of epiworld), building the same model;
- **epiworldpy 0.17.0-0** (Python wrapper of epiworld), building the same
  model;
- **Covasim 3.1.8** (Python), using its native disease progression and severe
  state as the hospitalization proxy;
- **Starsim 3.6.1** (Python), Covasim's successor, using its native SEIR
  disease with a hospital state added;
- **EoN 1.92** (Python), using a continuous-time SEIRH model run with the
  event-driven `fast_simple_contagion` algorithm;
- **epydemic 1.14.1** (Python), using a custom synchronous SEIRH model;
- **ixa 3.1.0** (Rust), using a custom synchronous SEIRH model driven by one
  plan per day on ixa's plan queue, its built-in contact network, and an
  indexed disease-status property; and
- **individual 0.1.19** (R, from mrc-ide), using a categorical state variable,
  its built-in transition processes, and an infection process over the
  network written for the benchmark. individual is not on CRAN, so the
  container installs the v0.1.19 release from GitHub.

The three epiworld packages are pinned to commits after the counting-sort network build
([UofUEpiBio/epiworld#274](https://github.com/UofUEpiBio/epiworld/issues/274)),
which no release includes yet and which still report version 0.17.0: epiworld
`092bad1` on master, epiworldR `1f5e63b` on main (in `.devcontainer/Dockerfile`),
and epiworldpy `4a1ee0b` on main (in `pyproject.toml`). Recorded versions do
not show the commit for epiworld and epiworldR, so results from an older
commit have to be cleared from the cache by hand when the pins change.

Each scenario lives in its own `scenario_NN/` folder with its report
(`README.qmd`, rendered to `README.md`), its parameters (`scenario.toml`), and
one runner per engine (`runners/`). Scenario 00 is the SEIRH baseline,
scenario 01 adds an all-or-nothing vaccine, scenario 02 adds epidemiological
outputs, and scenario 03 runs scenario 00 at 1,000,000 agents.
[scenarios.md](scenarios.md) explains how to add another.

The full design runs 100 replicates for 100 days at 10,000 and 100,000 agents
(`[study]` in `config.toml`). A scenario can replace those sizes and the
replicate count in the `[design]` table of its `scenario.toml`: scenario 03
runs 20 replicates at 1,000,000 agents.
Every engine receives the exact same cached Watts--Strogatz edge list at a
given population size. The graph has mean degree 10 and rewiring probability
0.05. Here “average density” is interpreted as the usual sparse-network
quantity, average degree: literal graph density would force degree (and memory)
to grow linearly with population size.

## Quick start

The benchmark is meant to run inside the container defined in
`.devcontainer/`. It pins R 4.5.1 with epiworldR 0.17.0.0 and individual
0.1.19, epiworld's 0.17.0 headers (both epiworld pins at the commits above),
Python 3.12 via uv (with epiworldpy and Starsim), Rust 1.98.0,
and Quarto 1.10.18, so every engine is built and timed on the same
toolchain. The only host prerequisite is [podman](https://podman.io/) (or
Docker: pass `CONTAINER=docker`). From this folder:

```sh
make container-image
make container-check
make container-smoke
make container-benchmark
make container-report
```

Every `container-TARGET` runs `make setup TARGET` in a fresh container with the
checkout bind-mounted at `/workspace`, so `cache/`, `results/`, and the
rendered report land in this folder. The Python environment (`/opt/venv`) and
the ixa and epiworld builds (`/opt/cargo-target` and `/opt/epiworld-build`, one
binary per scenario and engine) stay inside the
image and never touch a host `.venv/` or `target/`. Set `SCENARIOS` to run
only some scenarios, for example `SCENARIOS=scenario_01 make
container-benchmark`.

The same image is a development container: open the folder in VS Code (with
`"dev.containers.dockerPath": "podman"`) or another devcontainer client, and
`make setup`, `make check`, `make benchmark`, and so on work unchanged in its
shell.

Running natively is still possible given Python 3.12, uv, R with `epiworldR`,
`individual`, and `jsonlite`, a Rust toolchain, a C++17 compiler with zlib, and Quarto
(`make setup check smoke benchmark report`); `make setup` downloads epiworld's
headers into `.deps/`. Cache records carry the host OS and architecture in
their fingerprint, so native and container timings are never mixed.

The report target renders the data-free project overview (`README.md`) and
one GitHub-flavored report per scenario (`scenario_NN/README.md`, with PNG
figures in `scenario_NN/README_files/`), so the rendered results stay readable
directly in a pull request.

The smoke profile is deliberately tiny (1,000 agents, 10 days, one replicate)
and tests all nine integrations in every scenario. `make benchmark` launches
the full design: 1,800 runs for each of scenarios 00 to 02, and 180 for
scenario 03. It is safe to stop and restart: each successful replicate is written
atomically beneath `cache/results/`, and a later invocation only schedules
missing or stale results.

Useful targeted runs include:

```sh
# Preview work without launching a model
.venv/bin/python run.py --profile full --dry-run

# Run just one scenario or engine, or preflight one replicate at every full size
.venv/bin/python run.py --profile full --scenarios scenario_01
.venv/bin/python run.py --profile full --engines epiworldR
.venv/bin/python run.py --profile full --replicates 1

# Explicitly invalidate matching cached results
.venv/bin/python run.py --profile full --engines EoN --force
```

Do not delete `cache/` between runs. Records live at
`cache/results/<scenario>/<engine>/n<n>/`, and the cache key covers the shared
configuration, the scenario's parameters and runner sources, the engine
version, and the shared-network checksum. Changing one scenario's runners
invalidates only that scenario. `results/results.csv` (with a `scenario`
column), `results/scenarios.json`, and, for scenarios that report daily
series, `results/daily.csv` (medians across replicates) are rebuilt from all
successful cached records after each invocation. The report filters them to the full-profile
sizes and 100 days; `results/scenarios.json` records each scenario's sizes and
replicate counts.

## Resource policy

The default is intentionally conservative: one simulation subprocess at a
time, with OpenMP, BLAS, MKL, Accelerate, NumExpr, Numba, and Rcpp thread-count
variables all set to one. Network generation occurs once per size and is
excluded from engine timings.

Concurrency is opt-in through the `N_THREADS` environment variable, capped by
`resources.max_workers` in `config.toml` and by the host core count:

```sh
N_THREADS=3 make container-benchmark
```

`--workers` overrides `N_THREADS` when both are given. `N_THREADS` is not part
of the cache fingerprint, so changing it does not invalidate cached results.

Replicates running side by side contend for memory bandwidth and, on hybrid
CPUs, for performance cores, and the penalty differs by engine. Median
`simulate_seconds` at 100,000 agents, relative to a sequential run, measured
natively (before the container workflow and before ixa was added) on an 11-core
M3 Pro (5 performance + 6 efficiency), 12 replicates per cell:

| workers | epiworldR   | covasim     | EoN         |
|--------:|------------:|------------:|------------:|
| 2       | 0.98x       | 1.03x       | 1.00x       |
| 3       | 1.01x       | 1.20x       | 0.99x       |
| 4       | 1.06-1.30x  | 1.04-1.07x  | 0.97-0.99x  |
| 6       | 1.62x       | 1.42x       | 1.09x       |
| 8       | 1.42-2.17x  | 1.36-1.46x  | 1.13-1.27x  |

Ranges span repeated probes; cells without a range were measured once, and the
spread at four and eight workers shows these figures are sensitive to other
load on the host. The pattern is stable even so: up to three workers is free,
and beyond that the fastest engine degrades the most, because it is the one
most sensitive to being scheduled onto an efficiency core. That biases the
cross-engine comparison rather than just adding noise.

Three workers is the recommended setting on this host: roughly 3x throughput at
no measurable timing cost, staying within the five performance cores. Eight
workers bought only about 1.7x the throughput of three while distorting the
comparison. Use `N_THREADS=1` if you want timings collected under exactly the
documented sequential policy.

The container behaves the same way, more strongly. A 6-worker run of the full
design in the podman VM (10 vCPUs) inflated median epiworldR `simulate_seconds`
at 100,000 agents about 2.8x relative to a sequential run of the same model,
and ixa only 1.2-1.4x, roughly doubling the apparent ixa/epiworldR speed
ratio. The published results are collected with four workers
(`N_THREADS=4 make container-benchmark`), which keeps the full design,
including scenario 03's 1,000,000-agent runs, to a manageable run time. Every record in
them was collected at that concurrency, so they are comparable with each
other, but not with a sequential run.

Epidemiological outputs are unaffected by concurrency; only the timings are.

## What is timed

Each runner records:

- `read_seconds`: reading and parsing the shared edge list;
- `setup_seconds`: reading the edge list plus building what the engine can
  reuse across replicates on that network;
- `simulate_seconds`: everything the engine has to redo for each replicate,
  which is its simulation call plus, for ixa and Starsim, building a new
  context or `Sim` (neither can be run twice); and
- `total_seconds`: setup plus simulation inside the runner process.

Interpreter/package startup and shared graph generation are excluded. The
reports treat `simulate_seconds` as the primary outcome. They also show
build time (`setup_seconds - read_seconds`) and build + simulate, the time to
a first result once the edge list is in memory. Reading the file is left out
of both, because its cost depends on each language's file parsing rather than
on the engine. Each replicate runs in a fresh process, avoiding state
leakage and making failures independently resumable.


Relevant upstream documentation: [Covasim](https://docs.covasim.org/),
[Starsim](https://docs.starsim.org/),
[individual](https://mrc-ide.github.io/individual/),
[EoN generalized contagion](https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.fast_simple_contagion.html),
[epydemic synchronous dynamics](https://pyepydemic.readthedocs.io/en/latest/synchronousdynamics.html),
and [ixa](https://ixa.rs/).
