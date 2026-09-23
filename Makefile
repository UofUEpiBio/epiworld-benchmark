.PHONY: help setup check smoke benchmark report clean-cache container-image

# Inside the container, PYTHON and CARGO_TARGET_DIR come from the image.
PYTHON ?= .venv/bin/python
CARGO := cargo
IXA_MANIFESTS := $(sort $(wildcard scenario_*/runners/ixa/Cargo.toml))

# Restrict smoke/benchmark to some scenarios, e.g. SCENARIOS="scenario_01".
SCENARIOS ?=
SCENARIO_ARGS := $(if $(strip $(SCENARIOS)),--scenarios $(SCENARIOS),)

CONTAINER ?= podman
IMAGE ?= epiworld-benchmark

help:
	@echo "Available targets:"
	@echo "  setup           - Sync the Python environment and build every scenario's ixa runner"
	@echo "  check           - Run tests and check R package dependencies"
	@echo "  smoke           - Run a quick smoke test of the simulation"
	@echo "  benchmark       - Run the full benchmark simulation"
	@echo "  report          - Render the report using Quarto"
	@echo "  clean-cache     - Remove benchmark/cache manually if you really want to discard reusable runs."
	@echo ""
	@echo "smoke and benchmark run every scenario_* folder; set SCENARIOS to pick some."
	@echo ""
	@echo "Container targets (the documented way to run the benchmark):"
	@echo "  container-image - Build the $(IMAGE) image with $(CONTAINER)"
	@echo "  container-TARGET - Run 'make setup TARGET' in the container, e.g. container-benchmark"

setup:
	uv sync --frozen
	for manifest in $(IXA_MANIFESTS); do \
		$(CARGO) build --release --locked --manifest-path $$manifest || exit 1; \
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

README.md: README.qmd results/results.csv $(wildcard scenario_*/code_regions.yml)
	quarto render README.qmd

report: README.md

clean-cache:
	@echo "Remove benchmark/cache manually if you really want to discard reusable runs."

container-image:
	$(CONTAINER) build -t $(IMAGE) -f .devcontainer/Dockerfile .

container-%:
	$(CONTAINER) run --rm -v "$(CURDIR)":/workspace -w /workspace \
		-e N_THREADS -e SCENARIOS $(IMAGE) make setup $*
