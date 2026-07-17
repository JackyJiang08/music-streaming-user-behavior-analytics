PYTHON ?= python

.PHONY: help setup test lint data notebooks

help:  ## Show available targets
	@grep -E '^[a-z]+:.*##' $(MAKEFILE_LIST) | awk -F ':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## Install the locked dependency set (requirements.lock)
	$(PYTHON) -m pip install -r requirements.lock

test:  ## Run the unit-test suite
	$(PYTHON) -m pytest

lint:  ## Check style and formatting (ruff)
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check src scripts tests

data:  ## Rebuild the derived data assets (feature, survival, experiment tables)
	$(PYTHON) scripts/build_user_feature_table.py
	$(PYTHON) scripts/build_survival_table.py
	$(PYTHON) -m src.experiment_simulation

notebooks:  ## Execute all notebooks 01-10 in order (slow)
	@set -e; for nb in notebooks/0*.ipynb notebooks/10_*.ipynb; do \
		echo "=== $$nb"; \
		$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace "$$nb"; \
	done
