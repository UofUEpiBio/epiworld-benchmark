.PHONY: help setup check smoke benchmark collect profile report clean-cache container-image epiworld-runners fred-runner

# Inside the container, PYTHON and CARGO_TARGET_DIR come from the image.
PYTHON ?= .venv/bin/python
CARGO := cargo
IXA_MANIFESTS := $(sort $(wildcard scenario_*/runners/ixa/Cargo.toml))

# epiworld (C++) runners. The container provides the headers and the build
# directory; natively, the pinned commit is downloaded into .deps/ and each
# runner is built next to its source. Keep EPIWORLD_REF in step with
# EPIWORLD_SHA in the Dockerfile. The flags are the ones epiworldR is compiled with (R's -O2
# -DNDEBUG, and double rather than epiworld's default float), so the C++ and R
# runners differ only in the R layer.
# epiworld master at 0.17.1 (the automatic push/pull cost model, UofUEpiBio/epiworld#281).
EPIWORLD_REF := 04c4ad866b1866ca88332a888d941b395c96b8a0
EPIWORLD_INCLUDE ?= .deps/$(EPIWORLD_REF)/include
EPIWORLD_SOURCES := $(sort $(wildcard scenario_*/runners/epiworld/main.cpp))
EPIWORLD_CXXFLAGS := -std=c++17 -O2 -DNDEBUG -Depiworld_double=double

# FRED, for every scenario's FRED runner. It is a single shared
# binary, not compiled per scenario like epiworld/ixa above. The container
# bakes a prebuilt binary into the image at $FRED_HOME; natively, the pinned
# commit is downloaded into .deps/fred and built there. Keep FRED_REF in step
# with FRED_SHA in the Dockerfile.
FRED_REF := bd25f048f8e390ff697bdfa83a0bd3c2e19d6046
FRED_HOME ?= .deps/fred
FRED_BINARY := $(FRED_HOME)/bin/FRED

# Restrict smoke/benchmark to some scenarios, e.g. SCENARIOS="scenario_01".
SCENARIOS ?=
SCENARIO_ARGS := $(if $(strip $(SCENARIOS)),--scenarios $(SCENARIOS),)
# Extra run.py flags for smoke/benchmark, e.g. RUN_ARGS=--force.
RUN_ARGS ?=

CONTAINER ?= podman
IMAGE ?= epiworld-benchmark
# The host's CPU, which a container on macOS cannot see (see scripts/environment.py).
HOST_CPU := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null \
	|| sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -n 1)

help:
	@echo "Available targets:"
	@echo "  setup           - Sync the Python environment and build every scenario's ixa and epiworld runners"
	@echo "  check           - Run tests and check R package dependencies"
	@echo "  smoke           - Run a quick smoke test of the simulation"
	@echo "  benchmark       - Run the full benchmark simulation"
	@echo "  collect         - Rebuild results/*.csv from results/runs/ (e.g. after a git merge)"
	@echo "  profile         - Side measurements of epiworld and ixa behind analysis.md"
	@echo "  report          - Render the overview and every scenario report with Quarto"
	@echo "  clean-cache     - Remove benchmark/cache manually if you really want to discard reusable runs."
	@echo ""
	@echo "smoke and benchmark run every scenario_* folder; set SCENARIOS to pick some,"
	@echo "and RUN_ARGS for extra run.py flags (e.g. RUN_ARGS=--force)."
	@echo ""
	@echo "Container targets (the documented way to run the benchmark):"
	@echo "  container-image - Build the $(IMAGE) image with $(CONTAINER)"
	@echo "  container-TARGET - Run 'make setup TARGET' in the container, e.g. container-benchmark"

setup: epiworld-runners fred-runner
	uv sync --frozen
	for manifest in $(IXA_MANIFESTS); do \
		$(CARGO) build --release --locked --manifest-path $$manifest || exit 1; \
	done

$(EPIWORLD_INCLUDE):
	mkdir -p .deps/$(EPIWORLD_REF)
	curl -fsSL https://github.com/UofUEpiBio/epiworld/archive/$(EPIWORLD_REF).tar.gz \
		| tar -xz -C .deps/$(EPIWORLD_REF) --strip-components=1

epiworld-runners: | $(EPIWORLD_INCLUDE)
	for source in $(EPIWORLD_SOURCES); do \
		scenario=$${source%%/*}; \
		out="$${EPIWORLD_BUILD_DIR:-$$scenario/runners/epiworld/build}"; \
		mkdir -p "$$out"; \
		$(CXX) $(EPIWORLD_CXXFLAGS) -I$(EPIWORLD_INCLUDE) $$source -lz \
			-o "$$out/epiworld-$$(echo $$scenario | tr _ -)" || exit 1; \
	done

$(FRED_BINARY):
	mkdir -p $(FRED_HOME)
	curl -fsSL https://github.com/PublicHealthDynamicsLab/FRED/archive/$(FRED_REF).tar.gz \
		| tar -xz -C $(FRED_HOME) --strip-components=1
	echo $(FRED_REF) > $(FRED_HOME)/COMMIT
	# Upstream off-by-one: Date::setup_dates() writes past its date array.
	sed -i.orig 's/new date_t \[ Date::max_days \];/new date_t [ Date::max_days + 1 ];/' $(FRED_HOME)/src/Date.cc
	grep -q 'new date_t \[ Date::max_days + 1 \];' $(FRED_HOME)/src/Date.cc
	$(MAKE) -C $(FRED_HOME)/src FRED M64=

fred-runner: $(FRED_BINARY)

check:
	$(PYTHON) -m pytest
	for manifest in $(IXA_MANIFESTS); do \
		$(CARGO) test --release --locked --manifest-path $$manifest || exit 1; \
	done
	Rscript --vanilla -e 'stopifnot(requireNamespace("epiworldR", quietly = TRUE), requireNamespace("individual", quietly = TRUE), requireNamespace("jsonlite", quietly = TRUE))'

smoke:
	$(PYTHON) run.py --profile smoke $(SCENARIO_ARGS) $(RUN_ARGS)

benchmark:
	$(PYTHON) run.py --profile full $(SCENARIO_ARGS) $(RUN_ARGS)

collect:
	$(PYTHON) run.py --collect

# Run alone: concurrent work distorts the timings.
profile:
	PYTHON=$(PYTHON) bash analysis/profile.sh

# Always re-render: every scenario report reads the shared results and its
# own runner sources, which make cannot track cheaply.
report:
	quarto render README.qmd
	for report in scenario_*/README.qmd; do \
		quarto render $$report || exit 1; \
	done

clean-cache:
	@echo "Remove benchmark/cache manually if you really want to discard reusable runs."

container-image:
	$(CONTAINER) build -t $(IMAGE) -f .devcontainer/Dockerfile .

# BENCHMARK_IMAGE_ID and BENCHMARK_HOST_CPU go into each scenario's environment record.
container-%:
	$(CONTAINER) run --rm -v "$(CURDIR)":/workspace -w /workspace \
		-e N_THREADS -e SCENARIOS -e RUN_ARGS \
		-e BENCHMARK_IMAGE_ID=$$($(CONTAINER) image inspect -f '{{.Id}}' $(IMAGE)) \
		-e BENCHMARK_HOST_CPU="$(HOST_CPU)" \
		$(IMAGE) make setup $*
