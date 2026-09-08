# Network epidemic ABM speed benchmark

This folder contains a reproducible, resumable benchmark of four epidemic
simulation engines:

- **epiworldR 0.15.1.0** (R/C++), using a custom discrete-time SEIRH model;
- **Covasim 3.1.8** (Python), using its native symptom/severe bookkeeping and
  severe state as the hospitalization proxy, with waning immunity and
  transmission heterogeneity disabled;
- **EoN 1.92** (Python), using a continuous-time Gillespie SEIRH model; and
- **epydemic 1.14.1** (Python), using a custom synchronous SEIRH model.

The full design runs 100 replicates for 100 days at 10,000 and 100,000 agents.
Every engine receives the exact same cached Watts--Strogatz edge list at a
given population size. The graph has mean degree 10 and rewiring probability
0.05. Here “average density” is interpreted as the usual sparse-network
quantity, average degree: literal graph density would force degree (and memory)
to grow linearly with population size.

## Quick start

Prerequisites are Python 3.12, [uv](https://docs.astral.sh/uv/), R,
`epiworldR`, `jsonlite`, and Quarto. From this folder:

```sh
uv sync --frozen
make check
make smoke
make benchmark
make report
```

The report target produces GitHub-flavored `report.md` plus PNG figures in
`report_files/figure-commonmark/`, so the rendered results remain readable
directly in a pull request.

The smoke profile is deliberately tiny (1,000 agents, 10 days, one replicate)
and tests all four integrations. `make benchmark` launches the full 800-run
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
variables all set to one. `--workers 2` is available for an explicit opt-in,
and the harness hard-caps concurrency at two even on a many-core host. Network
generation occurs once per size and is excluded from engine timings.

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

## Epidemiological comparability

The common target is an SEIR process with an explicit hospitalization/recovery
branch, mean latent period 4 days, mean infectious period 7 days, 5% lifetime
hospitalization probability, 7-day hospital stay, 100 initial infections, and
an analytically targeted early-outbreak R0 of 2 on the degree-10 network.
Transmission is calibrated using edge transmissibility and the mean excess
degree (approximately degree minus one).

The analytic mapping initially gave epiworldR higher attack rates, so the full
cells use documented empirical transmission multipliers of 0.887 at 10,000
agents and 0.855 at 100,000 agents (`config.toml`). Each adjusted cell was rerun
for all 100 replicates. At 10,000 agents the median attack rates are now 0.3803
for epiworldR, 0.3984 for Covasim, 0.3774 for EoN, and 0.3958 for epydemic. At
100,000 agents they are 0.0620, 0.0690, 0.0611, and 0.0641, respectively. These
population-specific corrections prioritize comparable realized epidemic size;
the adjusted epiworldR cells should not be interpreted as having independently
validated $R_0$ values of exactly 2. Each multiplier is included in the cache
key and recorded in every adjusted result row.

epiworldR and epydemic use synchronous daily transitions. EoN implements the
same state graph with continuous-time competing hazards. Covasim retains its
native exposed, infectious, symptomatic, and severe bookkeeping, with severe
prevalence used as the hospitalization proxy. The runner disables Covasim's
waning-immunity/reinfection model and fixes both individual transmissibility
and viral load to one, while retaining that native severe-state path. Covasim
does not expose a public switch that removes the remaining generic
symptom/severity bookkeeping. Consequently, this is a **restricted,
representative framework-throughput comparison**, not proof that the engines
execute identical random processes. The report includes epidemic outcome checks
to make divergences visible.

The committed full-profile result artifacts predate this Covasim restriction.
They are retained as historical output; rerun `make benchmark` and `make
report` before using them to compare the revised design.

Relevant upstream documentation: [Covasim](https://docs.covasim.org/),
[EoN generalized contagion](https://epidemicsonnetworks.readthedocs.io/en/latest/functions/EoN.Gillespie_simple_contagion.html),
and [epydemic synchronous dynamics](https://pyepydemic.readthedocs.io/en/latest/synchronousdynamics.html).
