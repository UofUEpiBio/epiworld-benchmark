#!/usr/bin/env bash
# Side measurements behind docs/epiworld-ixa.md: where epiworld's and ixa's simulation
# time goes. Run inside the container, after a full benchmark has generated
# the networks, with nothing else running:
#
#   make container-profile
#
# Every runner is timed one process at a time (the benchmark runs four), for
# REPLICATES seeds, and the median simulate_seconds is printed. The variants:
#
#   transmission 1  the scenario as benchmarked
#   transmission 0  --transmission-multiplier 0: the seeds progress but nobody
#                   is infected, so what is left is the run's fixed cost
#   const-prob      the epiworld (C++) runner with the transmission
#                   probability set as a number instead of a parameter name
#   ixa-execute     the ixa runner timing execute() alone, as the benchmark
#                   did before it counted building the context as simulation
#
# Then the epiworld runner is run once more against a copy of the headers
# with phase timers (instrument_epiworld.py), one run per size.
set -euo pipefail

REPLICATES=${REPLICATES:-9}
INCLUDE=${EPIWORLD_INCLUDE:-/opt/epiworld/include}
FLAGS="-std=c++17 -O2 -DNDEBUG -Depiworld_double=double"
WORK=$(mktemp -d)
PYTHON=${PYTHON:-python3}

cp -r "$INCLUDE" "$WORK/instrumented"
"$PYTHON" analysis/instrument_epiworld.py "$WORK/instrumented"

median() { sort -g | awk '{ v[NR] = $1 } END { print v[int((NR + 1) / 2)] }'; }
simulate_ms() {
  "$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1]))['simulate_seconds'] * 1e3)" "$1"
}

for scenario in scenario_00 scenario_01; do
  source=$scenario/runners/epiworld/main.cpp
  sed 's/set_prob_infecting("Transmission rate")/set_prob_infecting(beta)/' "$source" \
    > "$WORK/const.cpp"
  grep -q 'set_prob_infecting(beta)' "$WORK/const.cpp"
  g++ $FLAGS -I"$INCLUDE" "$source" -lz -o "$WORK/epiworld"
  g++ $FLAGS -I"$INCLUDE" "$WORK/const.cpp" -lz -o "$WORK/const-prob"
  g++ $FLAGS -I"$WORK/instrumented" "$source" -lz -o "$WORK/instrumented-epiworld"
  cargo build --release --locked --quiet --manifest-path "$scenario/runners/ixa/Cargo.toml"
  target="${CARGO_TARGET_DIR:-$scenario/runners/ixa/target}"
  ixa="$target/release/ixa-${scenario/_/-}"
  # The same crate, renamed, with the timer moved down to execute().
  rm -rf "$WORK/ixa-execute" && cp -r "$scenario/runners/ixa" "$WORK/ixa-execute"
  rm -rf "$WORK/ixa-execute/target"
  sed -i "s/ixa-${scenario/_/-}/ixa-execute/" "$WORK/ixa-execute/Cargo.toml" "$WORK/ixa-execute/Cargo.lock"
  sed -i -e '/^    let simulate_started = Instant::now();$/d' \
    -e 's/^    context.execute();$/    let simulate_started = Instant::now();\n    context.execute();/' \
    "$WORK/ixa-execute/src/main.rs"
  [ "$(grep -c 'let simulate_started' "$WORK/ixa-execute/src/main.rs")" = 1 ]
  CARGO_TARGET_DIR="$target" cargo build --release --locked --quiet \
    --manifest-path "$WORK/ixa-execute/Cargo.toml"
  ixa_execute="$target/release/ixa-execute"

  extra=""
  if [ "$scenario" = scenario_01 ]; then extra="--vaccine-coverage 0.3 --vaccine-efficacy 0.8"; fi
  sizes="10000 100000 1000000"
  if [ "$scenario" = scenario_01 ]; then sizes="10000 100000"; fi

  index=0
  for n in $sizes; do
    network=$(ls cache/networks/watts-strogatz_n${n}_k10_p0.05_seed*.tsv.gz)
    # The benchmark's seeds for this size (config.toml base_seed).
    first_seed=$((8675309 + index * 100000 + 1))
    index=$((index + 1))
    args="--network $network --network-sha256 - --network-edges $((n * 5)) --n $n --days 100
      --replicate 1 --mean-degree 10 --target-r0 2.0 --initial-infected 100 --latent-days 4.0
      --infectious-days 7.0 --hospitalization-probability 0.05 --hospital-days 7.0 $extra
      --fingerprint - --engine-version - --output $WORK/record.json"
    for runner in epiworld const-prob ixa ixa-execute; do
      binary="$WORK/$runner"
      [ "$runner" = ixa ] && binary=$ixa
      [ "$runner" = ixa-execute ] && binary=$ixa_execute
      for multiplier in 1 0; do
        ms=$(for r in $(seq 0 $((REPLICATES - 1))); do
          $binary $args --seed $((first_seed + r)) --transmission-multiplier $multiplier
          simulate_ms "$WORK/record.json"
        done | median)
        printf '%s n=%-7s %-11s transmission %s  median simulate %7.2f ms\n' \
          "$scenario" "$n" "$runner" "$multiplier" "$ms"
      done
    done
    echo "$scenario n=$n epiworld phases (one run):"
    "$WORK/instrumented-epiworld" $args --seed "$first_seed" --transmission-multiplier 1 2>&1 \
      | sed 's/^/    /'
  done
done
rm -rf "$WORK"
