# PFE Makefile

PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTEST := $(PYTHON) -m pytest

.PHONY: help
help:
	@echo "PFE Development Commands"
	@echo "========================"
	@echo "make test        - Run unit tests (requires .venv)"
	@echo "make test-e2e    - Run end-to-end tests (requires .venv)"
	@echo "make test-all    - Run all tests (requires .venv)"
	@echo "make format      - Format code (if ruff/black available)"
	@echo "make lint        - Lint code (if ruff available)"
	@echo "make serve       - Start development server"
	@echo "make console     - Start interactive console"

.PHONY: check-venv
check-venv:
	@if [ ! -f .venv/bin/python ]; then \
		echo "Error: .venv/bin/python not found."; \
		echo "Please create a virtual environment first:"; \
		echo "  python3.11 -m venv .venv"; \
		echo "  source .venv/bin/activate"; \
		echo "  pip install -e ."; \
		exit 1; \
	fi

.PHONY: test
test: check-venv
	$(PYTEST) tests/ -v -m "not integration and not slow" --tb=short

.PHONY: test-e2e
test-e2e: check-venv
	$(PYTEST) tests/test_e2e_*.py -v --tb=short

.PHONY: test-all
test-all: check-venv
	$(PYTEST) tests/ -v --tb=short

.PHONY: format
format:
	-$(PYTHON) -m ruff format pfe-core pfe-cli pfe-server tests
	-$(PYTHON) -m black pfe-core pfe-cli pfe-server tests 2>/dev/null || true

.PHONY: lint
lint:
	-$(PYTHON) -m ruff check pfe-core pfe-cli pfe-server tests

.PHONY: serve
serve:
	$(PYTHON) -m pfe_server --port 8921 --workspace user_default

.PHONY: console
console:
	$(PYTHON) -m pfe_cli console
