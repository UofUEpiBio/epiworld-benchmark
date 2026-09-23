# Network epidemic ABM speed benchmark

This folder contains a reproducible, resumable benchmark of five epidemic
simulation engines:

- **epiworldR 0.15.1.0** (R/C++), using a custom discrete-time SEIRH model;
- **Covasim 3.1.8** (Python), using its native disease progression and severe
  state as the hospitalization proxy;
- **EoN 1.92** (Python), using a continuous-time SEIRH model run with the
  event-driven `fast_simple_contagion` algorithm;
- **epydemic 1.14.1** (Python), using a custom synchronous SEIRH model; and
- **ixa 3.1.0** (Rust), using a custom synchronous SEIRH model driven by one
  plan per day on ixa's plan queue, its built-in contact network, and an
  indexed disease-status property.

The full design runs 100 replicates for 100 days at 10,000 and 100,000 agents.
Every engine receives the exact same cached Watts--Strogatz edge list at a
given population size. The graph has mean degree 10 and rewiring probability
0.05. Here “average density” is interpreted as the usual sparse-network
quantity, average degree: literal graph density would force degree (and memory)
to grow linearly with population size.

## Quick start

The benchmark is meant to run inside the container defined in
`.devcontainer/`. It pins R 4.5.1 with epiworldR 0.15.1.0, Python 3.12 via uv,
Rust 1.98.0, and Quarto 1.10.18, so every engine is built and timed on the same
toolchain. The only host prerequisite is [podman](https://podman.io/) (or
Docker: pass `CONTAINER=docker`). From this folder:

```sh
make container-image
make container-check
make container-smoke
N_THREADS=6 make container-benchmark
make container-report
```

Every `container-TARGET` runs `make setup TARGET` in a fresh container with the
checkout bind-mounted at `/workspace`, so `cache/`, `results/`, and the
rendered report land in this folder. The Python environment (`/opt/venv`) and
the ixa build (`/opt/cargo-target`) stay inside the image and never touch a
host `.venv/` or `target/`.

The same image is a development container: open the folder in VS Code (with
`"dev.containers.dockerPath": "podman"`) or another devcontainer client, and
`make setup`, `make check`, `make benchmark`, and so on work unchanged in its
shell.

Running natively is still possible given Python 3.12, uv, R with `epiworldR`
and `jsonlite`, a Rust toolchain, and Quarto (`make setup check smoke
benchmark report`). Cache records carry the host OS and architecture in their
fingerprint, so native and container timings are never mixed.

The report target produces GitHub-flavored `README.md` plus PNG figures in
`README_files/figure-commonmark/`, so the rendered results remain readable
directly in a pull request.

The smoke profile is deliberately tiny (1,000 agents, 10 days, one replicate)
and tests all five integrations. `make benchmark` launches the full 1,000-run
design. It is safe to stop and restart: each successful replicate is written
atomically beneath `cache/results/`, and a later invocation only schedules
missing or stale results.

Useful targeted runs include:

```sh
# Preview work without launching a model
.venv/bin/python run.py --profile full --dry-run

# Run just one engine, or preflight one replicate at both full sizes
.venv/bin/python run.py --profile full --engines epiworldR
.venv/bin/python run.py --profile full --replicates 1

# Explicitly invalidate matching cached results
.venv/bin/python run.py --profile full --engines EoN --force
```

Do not delete `cache/` between runs. The cache key covers the configuration,
runner source, engine version, and shared-network checksum. `results/results.csv`
is rebuilt from all successful cached records after each invocation; the report
filters it to the full 10,000/100,000-agent, 100-day design.

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

Epidemiological outputs are unaffected by concurrency; only the timings are.

## What is timed

Each runner records:

- `setup_seconds`: reading the shared edge list and building engine objects;
- `simulate_seconds`: the engine's simulation call, including engine-native
  initialization that occurs inside that call; and
- `total_seconds`: setup plus simulation inside the runner process.

Interpreter/package startup and shared graph generation are excluded. The
Quarto report treats `simulate_seconds` as the primary outcome and presents
setup separately. Each replicate runs in a fresh process, avoiding state
leakage and making failures independently resumable.


Relevant upstream documentation: [Covasim](https://docs.covasim.org/),
[EoN generalized contagion](https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.fast_simple_contagion.html),
[epydemic synchronous dynamics](https://pyepydemic.readthedocs.io/en/latest/synchronousdynamics.html),
and [ixa](https://ixa.rs/).
