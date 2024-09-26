UV ?= uv
NPM ?= npm
PYTHON_PATHS := src tests scripts

.PHONY: help install lint format test build web-check browser-install e2e check demo hooks

help:
	@echo "install          Install locked Python and web dependencies"
	@echo "check            Run Python checks, tests, and both production builds"
	@echo "e2e              Run browser tests (browser-install required once)"
	@echo "demo             Build and launch the local demo"
	@echo "format           Format Python and apply safe lint fixes"
	@echo "hooks            Install optional pre-commit hooks"

install:
	$(UV) sync --locked
	$(NPM) --prefix app ci

lint:
	$(UV) run --locked ruff check $(PYTHON_PATHS)
	$(UV) run --locked ruff format --check $(PYTHON_PATHS)

format:
	$(UV) run --locked ruff check --fix $(PYTHON_PATHS)
	$(UV) run --locked ruff format $(PYTHON_PATHS)

test:
	$(UV) run --locked pytest

build:
	$(UV) build --no-sources

web-check:
	$(NPM) --prefix app run format:check
	$(NPM) --prefix app run lint
	$(NPM) --prefix app run test
	$(NPM) --prefix app run build

browser-install:
	cd app && npx playwright install chromium

e2e:
	$(NPM) --prefix app run test:e2e

check: lint test build web-check

demo:
	$(NPM) --prefix app run build
	$(UV) run --locked euv-diagnose demo

hooks:
	$(UV) run --locked pre-commit install
