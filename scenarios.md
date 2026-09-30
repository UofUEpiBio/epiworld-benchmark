# Adding a scenario

A scenario is one model that every engine implements. Scenarios are numbered
folders at the top of the repository. Each adds complexity to an earlier one,
so the report can show how run time and model code grow with the model.

The list of scenarios, with links to their reports, is in the
[project overview](README.md#scenarios).

## What a scenario folder holds

```
scenario_NN/
  README.qmd            # the report: spec, results, interpretation
  README.md             # README.qmd rendered by `make report`
  scenario.toml         # [scenario] [design] [disease] [intervention] [calibration]
  code_regions.yml      # line-count anchors for the report
  runners/
    epiworld.R          # epiworldR
    individual.R        # individual
    epiworld/main.cpp   # epiworld (C++), built by `make setup`
    run_epiworldpy.py   # one script per Python engine: run_epiworldpy.py,
    run_covasim.py      #   run_covasim.py, run_starsim.py, run_eon.py,
    run_starsim.py      #   run_epydemic.py; each holds that engine's whole
    run_eon.py          #   model
    run_epydemic.py
    runner_common.py    # argument parsing and result writing for the Python runners
    ixa/                # Cargo crate named ixa-scenario-NN
```

Everything else is shared: `config.toml` (study design, network, resources,
smoke profile), `run.py`, `scripts/`, and the report helpers in
`report/common.R`. `run.py` discovers every `scenario_*/scenario.toml`
automatically.

Each folder is self-contained on purpose. The argument parsing and result
writing are duplicated across scenarios so that each folder can be read as a
complete model. The report leaves that shared plumbing out of the line counts.

## Checklist

1. **Copy the scenario closest to the new one.**
   For example, `cp -R scenario_02 scenario_04`, then delete `runners/ixa/target`,
   `runners/epiworld/build`,
   `README.md`, and `README_files/` if you copied them.

2. **Rename the ixa crate.** In `runners/ixa/Cargo.toml`, set the package and
   `[[bin]]` names to `ixa-scenario-04`. Change the root package name in
   `Cargo.lock` to match, so that `cargo build --locked` still works. `run.py`
   expects the binary to be named `ixa-` plus the folder name, with `_`
   replaced by `-`.

3. **Edit `scenario.toml`.**
   - `[scenario]`: `name` and a one-line `description`. The report shows both.
   - `[disease]` and `[intervention]`: the model parameters. **Every key is
     passed to every runner as a `--kebab-case` flag**, so adding a parameter
     needs no change to `run.py`. Keys must be unique across both tables.
   - `[calibration]`: optional `<engine>_transmission_multiplier_<n>` factors.
     Say in a comment whether you copied or recalibrated them. `run.py`
     rejects keys that do not name a configured engine.

4. **Implement the change in every runner.**
   - Accept exactly the new flags: argparse in `runner_common.py`,
     `number("...")` or `integer("...")` in `epiworld.R`, `individual.R`, and
     `main.cpp`, and
     `Args` in `main.rs`. `tests/test_cache_and_network.py` checks that every
     parameter appears in each runner.
   - Use each engine's **own way** of expressing the feature: its built-in
     interventions, tools, compartments, or properties. The benchmark measures
     what a user of that engine would write, so do not work around the engine
     with generic code unless it offers nothing. When you have to, say why in
     the scenario README.
   - Keep the output record contract. Records must contain `final_susceptible`,
     `final_exposed`, `final_infected`, `final_hospitalized`,
     `final_recovered`, and `peak_hospitalized`, and the five final
     compartments must sum to `n`. Fold any new state into those five, the
     way scenario 01 counts protected agents as susceptible, and document the
     convention.
   - For new outcome fields (such as scenario 01's `vaccinated` and
     `vaccine_protected`), add the column to `fields` in
     `run.py:collect_results`. Older scenarios leave it blank. Note that
     editing `run.py` invalidates every scenario's cached results. Keep
     list-valued fields (such as scenario 02's daily series) out of
     `results.csv`; `write_daily_series()` in `run.py` shows how to publish a
     summary of them instead.
   - Time everything an engine has to redo for another replicate on the same
     network. The simulation timer covers each run's initial conditions
     (seeding, vaccination, and the like), because epiworld sets them up
     inside `run()`. ixa does this in plans at time 0; the Python runners and
     `individual.R` start their timer before building the initial state. An
     engine object that can run only once is rebuilt for every replicate, so
     it is timed too: ixa's context (population, network, and index) and
     Starsim's `Sim`, including `sim.init()`.
   - Record `read_seconds`, the time to read the edge list, right after the
     edge count is checked. The reports subtract it from setup to show build
     time.
   - Keep randomness seeded from `--seed`. Seeds do not depend on the
     scenario, so the report can compare scenarios replicate by replicate.
   - Add Rust unit tests for the new behaviour to `main.rs`.

5. **Update `code_regions.yml`.** Add regions covering the new model code.
   Leave out argument parsing, edge-file reading, timing, and result writing.
   The report fails if an anchor is not found. This file is not part of the
   cache fingerprint, so you can adjust it after a run without rerunning
   anything.

6. **Write the report, `README.qmd`.** Start from the previous scenario's
   report. It should describe the model change, include a parameter table,
   state the output convention, and have one bullet per engine describing how
   it is implemented. The tables and figures come from `report/common.R`:
   call `load_benchmark("scenario_04")`, then the `table_*` and `plot_*`
   helpers. To compare with an earlier scenario, pass it to
   `table_time_versus()` and `table_code_effort()`. Write the interpretation
   specific to the scenario, and add a row for it to the scenario table in
   the top-level `README.qmd`.

7. **Check correctness before timing anything.**
   ```sh
   make container-check
   SCENARIOS=scenario_04 make container-smoke
   ```
   Then try an extreme setting where the answer is known. For scenario 01,
   coverage and efficacy of 1 must leave only the seed cases infected. Then
   run a small full-size sample and check that the engines' attack rates
   agree:
   ```sh
   .venv/bin/python run.py --profile full --scenarios scenario_04 --replicates 5
   ```
   If one engine is far slower than expected, profile it before accepting
   the number. Scenario 01 found a quadratic cost in epiworldR's
   `distribute_tool_to_set()` this way.

8. **Run and report.**
   ```sh
   SCENARIOS=scenario_04 make container-benchmark
   make container-report
   ```
   A scenario runs at the sizes and replicate count in `[study]` in
   `config.toml`, unless its `scenario.toml` has a `[design]` table with its
   own `population_sizes` and `replicates`, as scenario 03 does. A new size
   needs transmission multipliers for that size in `[calibration]`, and a
   size index: its network and seeds follow its position after the
   `[study]` sizes, in scenario order.
   Leave `N_THREADS` unset (one replicate at a time), as the published
   results do, so the new scenario's timings match the others'. `make report`
   renders the overview and every `scenario_*/README.qmd`. Check that the
   prose in the new report still matches its numbers.

## Caching

Each full-profile replicate is committed as one record at
`results/runs/<scenario>/<engine>/n<n>/replicate-NNN.json` (smoke records stay
in the untracked `cache/results/`), and it is reused whenever its fingerprint
matches. The fingerprint is a SHA-256 of the replicate's identity: the
scenario's own `scenario.toml` and `runners/` (a hash of their contents), the
engine and its version, the population size, days, replicate, and seed, the
network's checksum (of its decompressed edge list, so it does not depend on
the zlib build that compressed it) and edge count, the model parameters and transmission
multiplier, the operating system and architecture, and the record format.
Editing one scenario's runners therefore never invalidates another scenario's
results. Editing `run.py`, `scripts/`, or anything else outside the scenario
invalidates nothing on its own: `config.toml` matters only through the
parameters, sizes, and seeds it feeds into the identity, and a change to how
records are produced must bump `RUNNER_FORMAT_VERSION` in `run.py`. Every
record carries that `format_version`, and a record in an older format is never
reused, even when a scenario accepts published results.

Where a scenario's results were produced is recorded separately, in
`results/environments/<scenario>.json`, and by an `environment_id` in each
replicate. Neither is part of the fingerprint, so they never invalidate cached
results. A run that only reads the cache leaves the scenario's record as it was;
see "Environment records" in [`setup.md`](setup.md).
