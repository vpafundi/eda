.PHONY: install install-s3 kernel test lint fmt run notebook clean

# Ambiente base (core + ml + notebook)
install:
	uv sync

# SageMaker / parquet su S3
install-s3:
	uv sync --group s3

# Kernel Jupyter per il notebook
kernel:
	uv run python -m ipykernel install --user --name eda-aic --display-name "Python (eda-aic)"

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

# Run EDA headless: make run DATA=/path/to/segments.parquet [TAG=nov2026]
DATA ?= $(EDA_DATA_PATH)
TAG ?= manual
run:
	uv run eda-run --data "$(DATA)" --tag "$(TAG)"

notebook:
	uv run jupyter lab notebooks/

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
