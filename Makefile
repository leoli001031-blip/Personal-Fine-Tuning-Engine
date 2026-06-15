# PFE Makefile

PYTHON := $(shell if [ -f .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PYTEST := $(PYTHON) -m pytest
FAST_MARKERS := not integration and not slow
UNIT_IGNORE := --ignore-glob='tests/*surface*.py' --ignore-glob='tests/test_e2e_*.py' --ignore-glob='tests/test_*_e2e.py'
SURFACE_TESTS := tests/*surface*.py
E2E_TESTS := tests/test_e2e_*.py tests/test_*_e2e.py
AUDIT_REPORT ?= /tmp/pfe-release-evidence-audit.json
PERF_REPORT ?= /tmp/pfe-release-perf-report.json
REMOTE_EVIDENCE_REPORT ?= /tmp/pfe-github-actions-release-evidence.json
BUNDLE_REPORT ?= /tmp/pfe-release-evidence-bundle.json
REMOTE_EVIDENCE_MARKDOWN ?= /tmp/pfe-remote-release-evidence.md

.PHONY: help
help:
	@echo "PFE Development Commands"
	@echo "========================"
	@echo "make test          - Run fast non-integration/non-slow tests (requires .venv)"
	@echo "make test-unit     - Run fast unit/core tests, excluding surface/e2e files"
	@echo "make test-surface  - Run fast CLI/HTTP/browser surface tests"
	@echo "make test-e2e-mock - Run lightweight mock e2e tests only"
	@echo "make smoke-first-run - Run the isolated first-run CLI smoke"
	@echo "make smoke-auto-train-queue - Run the isolated auto-train queue smoke"
	@echo "make smoke-real-local-readiness - Run the real-local readiness smoke"
	@echo "make smoke-studio-model-path - Run the Studio model-path/API handoff smoke"
	@echo "make smoke-server-live - Launch a temporary live server and probe HTTP surfaces"
	@echo "make smoke-dashboard-console-live - Probe dashboard and Studio live surfaces"
	@echo "make smoke-browser-ui-live - Optional Playwright browser smoke for dashboard/Studio UI"
	@echo "make smoke-studio-first-launch - Clean-install and launch Studio from the console script"
	@echo "make smoke-real-local-happy - Opt-in real local model happy path (set PFE_REAL_LOCAL_MODEL)"
	@echo "make smoke-beta - Run the beta-ready smoke chain"
	@echo "make smoke-release-strict - Run beta smoke plus required browser/model gates"
	@echo "make soak-release - Run a bounded live release soak over server/dashboard/queue/daemon"
	@echo "make benchmark-release - Record release timing and memory baselines"
	@echo "make release-local-evidence - Run local release gate and write evidence reports"
	@echo "make audit-release-evidence - Audit local release evidence and remaining remote gap"
	@echo "make audit-release-evidence-report - Write a JSON release evidence audit report"
	@echo "make bundle-release-evidence - Validate and summarize release evidence JSON reports"
	@echo "make record-remote-release-evidence - Record latest successful GitHub Actions run evidence"
	@echo "make render-remote-release-evidence - Render remote Actions evidence as Markdown"
	@echo "make test-e2e      - Run all e2e tests (may require services)"
	@echo "make test-all      - Run all tests (requires .venv)"
	@echo "make format        - Format code (if ruff/black available)"
	@echo "make lint          - Lint code (if ruff available)"
	@echo "make studio        - Start PFE Studio and open the browser"
	@echo "make serve         - Start development server"
	@echo "make console       - Start interactive console"

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
	$(PYTEST) tests/ -v -m "$(FAST_MARKERS)" --tb=short

.PHONY: test-unit
test-unit: check-venv
	$(PYTEST) tests/ -v -m "$(FAST_MARKERS) and not e2e" $(UNIT_IGNORE) --tb=short

.PHONY: test-surface
test-surface: check-venv
	$(PYTEST) $(SURFACE_TESTS) -v -m "$(FAST_MARKERS)" --tb=short

.PHONY: test-e2e-mock
test-e2e-mock: check-venv
	$(PYTEST) $(E2E_TESTS) -v -m "$(FAST_MARKERS)" --tb=short

.PHONY: smoke-first-run
smoke-first-run: check-venv
	$(PYTHON) tools/first_run_smoke.py

.PHONY: smoke-auto-train-queue
smoke-auto-train-queue: check-venv
	$(PYTHON) tools/first_run_smoke.py --stop-after queue

.PHONY: smoke-real-local-readiness
smoke-real-local-readiness: check-venv
	$(PYTHON) tools/real_local_readiness_smoke.py

.PHONY: smoke-studio-model-path
smoke-studio-model-path: check-venv
	$(PYTHON) tools/studio_model_path_smoke.py

.PHONY: smoke-server-live
smoke-server-live: check-venv
	$(PYTHON) tools/server_live_smoke.py

.PHONY: smoke-dashboard-console-live
smoke-dashboard-console-live: check-venv
	$(PYTHON) tools/dashboard_console_live_smoke.py

.PHONY: smoke-browser-ui-live
smoke-browser-ui-live: check-venv
	$(PYTHON) tools/browser_ui_live_smoke.py

.PHONY: smoke-studio-first-launch
smoke-studio-first-launch: check-venv
	$(PYTHON) tools/studio_first_launch_smoke.py

.PHONY: smoke-real-local-happy
smoke-real-local-happy: check-venv
	$(PYTHON) tools/real_local_happy_path_smoke.py

.PHONY: smoke-beta
smoke-beta: smoke-first-run smoke-auto-train-queue smoke-real-local-readiness smoke-studio-model-path smoke-server-live smoke-dashboard-console-live

.PHONY: smoke-release-strict
smoke-release-strict: smoke-beta
	$(PYTHON) tools/studio_first_launch_smoke.py
	$(PYTHON) tools/browser_ui_live_smoke.py --strict
	$(PYTHON) tools/real_local_happy_path_smoke.py --strict

.PHONY: soak-release
soak-release: check-venv
	$(PYTHON) tools/release_soak_smoke.py

.PHONY: benchmark-release
benchmark-release: check-venv
	$(PYTHON) tools/release_perf_benchmark.py --report-path $(PERF_REPORT)

.PHONY: release-local-evidence
release-local-evidence: check-venv
	$(MAKE) test-e2e-mock
	$(MAKE) smoke-release-strict
	$(MAKE) benchmark-release PERF_REPORT=$(PERF_REPORT)
	$(MAKE) audit-release-evidence-report AUDIT_REPORT=$(AUDIT_REPORT)
	$(MAKE) bundle-release-evidence PERF_REPORT=$(PERF_REPORT) AUDIT_REPORT=$(AUDIT_REPORT) REMOTE_EVIDENCE_REPORT=$(REMOTE_EVIDENCE_REPORT) BUNDLE_REPORT=$(BUNDLE_REPORT)

.PHONY: audit-release-evidence
audit-release-evidence: check-venv
	$(PYTHON) tools/release_evidence_audit.py

.PHONY: audit-release-evidence-report
audit-release-evidence-report: check-venv
	$(PYTHON) tools/release_evidence_audit.py --report-path $(AUDIT_REPORT)

.PHONY: record-remote-release-evidence
record-remote-release-evidence: check-venv
	$(PYTHON) tools/github_actions_release_evidence.py --require-success --output-path $(REMOTE_EVIDENCE_REPORT)

.PHONY: bundle-release-evidence
bundle-release-evidence: check-venv
	$(PYTHON) tools/release_evidence_bundle.py --perf-report $(PERF_REPORT) --audit-report $(AUDIT_REPORT) --remote-evidence-report $(REMOTE_EVIDENCE_REPORT) --output-path $(BUNDLE_REPORT)

.PHONY: render-remote-release-evidence
render-remote-release-evidence: check-venv
	$(PYTHON) tools/render_remote_release_evidence.py --remote-evidence-report $(REMOTE_EVIDENCE_REPORT) --bundle-report $(BUNDLE_REPORT) --output-path $(REMOTE_EVIDENCE_MARKDOWN) --require-success

.PHONY: test-e2e
test-e2e: check-venv
	$(PYTEST) $(E2E_TESTS) -v --tb=short

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

.PHONY: studio
studio: check-venv
	$(PYTHON) -m pfe_server --port 8921 --workspace user_default

.PHONY: serve
serve: check-venv
	$(PYTHON) -m pfe_server --port 8921 --workspace user_default --no-open

.PHONY: console
console:
	$(PYTHON) -m pfe_cli console
