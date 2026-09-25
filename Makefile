# The same commands CI runs, so a green `make check` means a green pipeline.
.PHONY: help venv build check fmt lint types test test-unit test-live clean

PY := .venv/bin/python

help:
	@echo "make venv        create .venv and install dev dependencies"
	@echo "make build       build and install the native extension (editable)"
	@echo "make check       everything CI runs"
	@echo "make fmt         format Python and Rust in place"
	@echo "make test        full test suite (live tests skip without credentials)"
	@echo "make test-unit   pure-Python tests, no Rust toolchain needed"
	@echo "make test-live   live Databricks suite (needs a PAT; see docs/testing.md)"
	@echo "make clean       remove build artifacts"

venv:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) maturin -e ".[dev,pyarrow]"

build:
	$(PY) -m maturin develop --uv

check: lint types test
	cargo fmt --all --check
	cargo clippy --all-targets -- -D warnings
	cargo test --all

fmt:
	ruff check . --fix
	ruff format .
	cargo fmt --all

lint:
	ruff check .
	ruff format --check .

types:
	mypy

test:
	$(PY) -m pytest -q

test-unit:
	$(PY) -m pytest tests/unit -q

test-live:
	$(PY) -m tests.live.preflight
	$(PY) -m pytest tests/live -v

clean:
	cargo clean
	rm -rf dist build .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
