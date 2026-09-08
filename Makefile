.PHONY: setup check smoke benchmark report clean-cache 

PYTHON := .venv/bin/python

help:
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and sync the environment"
	@echo "  check         - Run tests and check R package dependencies"
	@echo "  smoke         - Run a quick smoke test of the simulation"
	@echo "  benchmark     - Run the full benchmark simulation"
	@echo "  report        - Render the report using Quarto"
	@echo "  clean-cache   - Remove benchmark/cache manually if you really want to discard reusable runs."

setup:
	uv sync --frozen

check:
	$(PYTHON) -m pytest
	Rscript --vanilla -e 'stopifnot(requireNamespace("epiworldR", quietly = TRUE), requireNamespace("jsonlite", quietly = TRUE))'

smoke:
	$(PYTHON) run.py --profile smoke

benchmark:
	$(PYTHON) run.py --profile full

README.md: README.qmd
	quarto render README.qmd

report: README.md

clean-cache:
	@echo "Remove benchmark/cache manually if you really want to discard reusable runs."
