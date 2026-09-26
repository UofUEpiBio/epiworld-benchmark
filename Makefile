.PHONY: help setup check smoke benchmark report clean-cache container-image epiworld-runners

# Inside the container, PYTHON and CARGO_TARGET_DIR come from the image.
PYTHON ?= .venv/bin/python
CARGO := cargo
IXA_MANIFESTS := $(sort $(wildcard scenario_*/runners/ixa/Cargo.toml))

# epiworld (C++) runners. The container provides the headers and the build
# directory; natively, the pinned release is downloaded into .deps/ and each
# runner is built next to its source. Keep EPIWORLD_TAG in step with the
# Dockerfile. The flags are the ones epiworldR is compiled with (R's -O2
# -DNDEBUG, and double rather than epiworld's default float), so the C++ and R
# runners differ only in the R layer.
EPIWORLD_TAG := epiworld-v0.17.0
EPIWORLD_INCLUDE ?= .deps/$(EPIWORLD_TAG)/include
EPIWORLD_SOURCES := $(sort $(wildcard scenario_*/runners/epiworld/main.cpp))
EPIWORLD_CXXFLAGS := -std=c++17 -O2 -DNDEBUG -Depiworld_double=double

# Restrict smoke/benchmark to some scenarios, e.g. SCENARIOS="scenario_01".
SCENARIOS ?=
SCENARIO_ARGS := $(if $(strip $(SCENARIOS)),--scenarios $(SCENARIOS),)

CONTAINER ?= podman
IMAGE ?= epiworld-benchmark

help:
	@echo "Available targets:"
	@echo "  setup           - Sync the Python environment and build every scenario's ixa and epiworld runners"
	@echo "  check           - Run tests and check R package dependencies"
	@echo "  smoke           - Run a quick smoke test of the simulation"
	@echo "  benchmark       - Run the full benchmark simulation"
	@echo "  report          - Render the overview and every scenario report with Quarto"
	@echo "  clean-cache     - Remove benchmark/cache manually if you really want to discard reusable runs."
	@echo ""
	@echo "smoke and benchmark run every scenario_* folder; set SCENARIOS to pick some."
	@echo ""
	@echo "Container targets (the documented way to run the benchmark):"
	@echo "  container-image - Build the $(IMAGE) image with $(CONTAINER)"
	@echo "  container-TARGET - Run 'make setup TARGET' in the container, e.g. container-benchmark"

setup: epiworld-runners
	uv sync --frozen
	for manifest in $(IXA_MANIFESTS); do \
		$(CARGO) build --release --locked --manifest-path $$manifest || exit 1; \
	done

$(EPIWORLD_INCLUDE):
	mkdir -p .deps/$(EPIWORLD_TAG)
	curl -fsSL https://github.com/UofUEpiBio/epiworld/archive/refs/tags/$(EPIWORLD_TAG).tar.gz \
		| tar -xz -C .deps/$(EPIWORLD_TAG) --strip-components=1

epiworld-runners: | $(EPIWORLD_INCLUDE)
	for source in $(EPIWORLD_SOURCES); do \
		scenario=$${source%%/*}; \
		out="$${EPIWORLD_BUILD_DIR:-$$scenario/runners/epiworld/build}"; \
		mkdir -p "$$out"; \
		$(CXX) $(EPIWORLD_CXXFLAGS) -I$(EPIWORLD_INCLUDE) $$source -lz \
			-o "$$out/epiworld-$$(echo $$scenario | tr _ -)" || exit 1; \
	done

check:
	$(PYTHON) -m pytest
	for manifest in $(IXA_MANIFESTS); do \
		$(CARGO) test --release --locked --manifest-path $$manifest || exit 1; \
	done
	Rscript --vanilla -e 'stopifnot(requireNamespace("epiworldR", quietly = TRUE), requireNamespace("jsonlite", quietly = TRUE))'

smoke:
	$(PYTHON) run.py --profile smoke $(SCENARIO_ARGS)

benchmark:
	$(PYTHON) run.py --profile full $(SCENARIO_ARGS)

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

container-%:
	$(CONTAINER) run --rm -v "$(CURDIR)":/workspace -w /workspace \
		-e N_THREADS -e SCENARIOS $(IMAGE) make setup $*
