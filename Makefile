.PHONY: setup check smoke benchmark report clean-cache

PYTHON := .venv/bin/python

setup:
	uv sync --frozen

check:
	$(PYTHON) -m pytest
	Rscript --vanilla -e 'stopifnot(requireNamespace("epiworldR", quietly = TRUE), requireNamespace("jsonlite", quietly = TRUE))'

smoke:
	$(PYTHON) run.py --profile smoke

benchmark:
	$(PYTHON) run.py --profile full

report:
	quarto render report.qmd

clean-cache:
	@echo "Remove benchmark/cache manually if you really want to discard reusable runs."
