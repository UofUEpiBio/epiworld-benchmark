# Running the benchmark

[Back to the project overview](README.md)

How to build, run, and report the benchmark. The [methods](docs/methods.md)
explain what it measures and why it runs the way it does.

## Quick start

The benchmark is meant to run inside the container defined in
`.devcontainer/`. It pins R 4.5.1, Python 3.12 via uv, Rust 1.98.0, Julia
1.13.1, and Quarto 1.10.18, with every engine at the version in
[Engines](docs/methods.md#engines), so every engine is built and timed on the
same toolchain. The only host prerequisite is [podman](https://podman.io/) (or
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

Every pull request and push to `main` runs `make setup check smoke` on GitHub
Actions (`.github/workflows/tests.yml`), inside this same image on arm64. The
workflow tags the image with a hash of the files the Dockerfile copies in,
builds and pushes it to `ghcr.io/uofuepibio/epiworld-benchmark` only when one
of them changes, and pulls it otherwise; `main` also tags it `latest`. To skip
the local build, pull it and point the Makefile at it:

```sh
podman pull ghcr.io/uofuepibio/epiworld-benchmark:latest
make container-check IMAGE=ghcr.io/uofuepibio/epiworld-benchmark:latest
```

The same image is a development container: open the folder in VS Code (with
`"dev.containers.dockerPath": "podman"`) or another devcontainer client, and
`make setup`, `make check`, `make benchmark`, and so on work unchanged in its
shell.

Running natively is still possible given Python 3.12, uv, R with `epiworldR`,
`individual`, `ABM`, `EpiModel`, and `jsonlite`, a Rust toolchain, a C++17 compiler with zlib, Quarto, and,
for each run's process peak memory, GNU `time` (Linux only)
(`make setup check smoke benchmark report`); `make setup` downloads epiworld's
headers into `.deps/`. Cache records carry the host OS and architecture in
their fingerprint, so native and container timings are never mixed.

The report target renders the overview (`README.md`), the methods and results
(`docs/methods.md`, `docs/results.md`), and one report per scenario
(`scenario_NN/README.md`), as GitHub-flavored Markdown with PNG figures, so
the rendered results stay readable directly in a pull request.

The smoke profile is deliberately tiny (1,000 agents, 10 days, one replicate)
and tests all thirteen integrations in every scenario. `make benchmark` launches
the full design: 2,200 runs for each of scenarios 00 to 02, and 220 for each
of scenarios 03 and 04. It is safe to stop and restart: each successful replicate is written
atomically beneath `results/runs/`, and a later invocation only schedules
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

Each full-profile replicate is one JSON record at
`results/runs/<scenario>/<engine>/n<n>/replicate-NNN.json`, committed with the
results. A replicate whose record exists with the fingerprint the run expects
is skipped, so a fresh clone, or another machine, reruns only what is missing
or stale, not what has been published. Runs from several machines combine by
committing their records; they touch different files, so git merges them
cleanly. Smoke records are throwaway and stay in `cache/results/`, next to the
generated networks, which are not committed. The fingerprint is described in
"Caching" in [`scenarios.md`](scenarios.md#caching); it includes the
operating system and architecture, so a machine of another architecture
reruns everything.

`results/results.csv` (with a `scenario` column), `results/scenarios.json`,
and, for scenarios that report daily series, `results/daily.csv` (medians
across replicates) are derived from `results/runs/` alone after each
invocation, so they are the same on every clone. After merging branches that
both changed records, regenerate them with `make collect` (or
`run.py --collect`) rather than resolving conflicts by hand. Records of a
replicate the design no longer includes (for example, after lowering a
replicate count) are not removed automatically; delete them from
`results/runs/` before collecting. The report filters the results to the
full-profile sizes and 100 days; `results/scenarios.json` records each
scenario's sizes and replicate counts.

## Pins

Every engine is pinned. The epiworld family is pinned to commits on its
default branches: epiworld (`EPIWORLD_SHA` in
[`.devcontainer/Dockerfile`](.devcontainer/Dockerfile), kept in step with
`EPIWORLD_REF` in the [Makefile](Makefile)), epiworldR (`EPIWORLDR_SHA` in the
Dockerfile), and epiworldpy (in [`pyproject.toml`](pyproject.toml) and
`uv.lock`). FRED (`FRED_SHA`), individual (`INDIVIDUAL_SHA`), ABM (its CRAN version), and EpiModel (the
image's dated CRAN snapshot) are pinned in
the Dockerfile, the Python engines in `uv.lock`, Agents.jl in
`julia/Manifest.toml`, and ixa in each runner's `Cargo.lock`. Each engine's
recorded version ends in the commit it was built from where the version
number does not change with it (for example `0.17.1+g04c4ad8`), so moving a
pin invalidates that engine's published results. Natively, epiworldR's
commit is unknown and its version is recorded without it.

## Environment records

Timings only mean something next to the machine that produced them, so a run
that executes at least one replicate of a scenario writes
`results/environments/<scenario>.json` (via `scripts/environment.py`). It holds:

- hardware: CPU model, logical and physical cores, total memory, and the cgroup
  CPU and memory limits of the container, if any (`null` when unlimited). On
  macOS the container runs in a virtual machine that hides the CPU model, so
  `make container-TARGET` also records the host's as `host_cpu_model`, and
  cores and memory are those of the virtual machine;
- OS: platform, kernel, architecture, and whether it runs in a container;
- the container image ID (`make container-TARGET` passes it as
  `BENCHMARK_IMAGE_ID`; `null` natively);
- toolchain versions: Python, R, Julia, Rust, the C++ compiler, and Quarto;
- the pinned commits of epiworld, epiworldR, FRED, and individual (the image
  records them as environment variables; a native run reads epiworld's and
  FRED's from `.deps/`, and epiworldR's and individual's are `null`);
- the engine versions, this checkout's commit and whether it has uncommitted
  changes (files under `results/` are ignored), the profile, the harness
  concurrency, and the thread variables set for the runners;
- when the run started and finished, and how many replicates it executed,
  found in the cache, and failed.

The container cannot see the host's CPU model or its own image ID, so
`make container-TARGET` reads both on the host and passes them in as
`BENCHMARK_HOST_CPU` (from `sysctl -n machdep.cpu.brand_string` on macOS, or
the `model name` in `/proc/cpuinfo` on Linux) and `BENCHMARK_IMAGE_ID`. A
container started any other way, with `podman run`, `docker run`, or as a
devcontainer, records both as `null`. Inside a devcontainer on macOS, the
record then names only the CPU's vendor ("Apple (model not reported)"), not
the model the timings ran on; on a Linux host, the container's own
`cpu_model` is still the real one. Both fields are part of `environment_id`, so the
replicates of such a run also get a different id from those run through
`make`, and the scenario report shows its mixed-environment warning. Run
published benchmarks through `make container-TARGET`, or pass both variables
yourself.

A field that cannot be read is `null`. `environment_id` is a SHA-256 of the
hardware, OS, image, toolchains, pins, and engine versions, so two runs on the
same setup share it whatever their dates, commit, or concurrency. It is stored in each
replicate's cache record and appears as a column of `results/results.csv`; a
scenario report warns when its replicates carry more than one id. The id covers
the engine versions of the invocation, so a run restricted with `--engines` gets
a different one from a full run. Replicates cached before ids were recorded have
none.

A scenario whose replicates were all cached keeps its previous record, and a
run of one scenario never touches another's. `results/run-manifest.json` remains
the log of the latest invocation (profile, requested scenarios and engines,
counts, workers, Python version).

## Concurrency

Replicates run one at a time by default, with native math libraries pinned to
one thread; published results must be collected this way (see
[Execution protocol](docs/methods.md#execution-protocol) for why).
Concurrency is opt-in for smoke and exploratory runs, through the `N_THREADS`
environment variable, capped by `resources.max_workers` in `config.toml` and
by the host core count:

```sh
N_THREADS=3 make container-benchmark
```

`--workers` overrides `N_THREADS` when both are given. `N_THREADS` is not part
of the fingerprint, so changing it does not invalidate published results. A
scenario's `[design]` table can set `workers = 1`, as scenarios 03 and 04 do,
to stay sequential whatever `N_THREADS` says.
