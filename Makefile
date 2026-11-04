# Self-improving speculative decoding fleet
#
# Two run modes, deliberately separate:
#   fleet-*   containerized fleet (CPU inference) -- the control plane,
#             hot-swap, canary and rollback end to end
#   bench-*   host GPU runs -- the throughput and correctness numbers
# See README "Which numbers come from where".

PY       := .venv/bin/python
PYTEST   := .venv/bin/pytest
RUN      := PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PY)
COMPOSE  := docker compose

.DEFAULT_GOAL := help
.PHONY: help setup db db-down schema test test-fast lint prompts \
        bench-smoke bench bench-report precision fleet-build fleet-up \
        fleet-down fleet-logs fleet-status fleet-traffic clean nuke

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv and install dependencies (Python 3.12)
	uv python install 3.12
	uv sync --extra dev --extra quant

db: ## Start Postgres (the registry) in Docker
	$(COMPOSE) up -d postgres
	@until $(COMPOSE) exec -T postgres pg_isready -U fleet -d fleet >/dev/null 2>&1; \
	  do sleep 1; done; echo "postgres ready on localhost:5433"

db-down: ## Stop Postgres
	$(COMPOSE) stop postgres

schema: db ## Apply the registry schema and register a cold-start version
	$(RUN) -m fleet.cli bootstrap

test: db ## Full test suite (needs Postgres)
	PYTHONPATH=src $(PYTEST) -q

test-fast: ## Tests that need no database
	PYTHONPATH=src $(PYTEST) -q tests/test_greedy_equivalence.py tests/test_drift.py \
	  tests/test_drafters.py

lint: ## Ruff check + format check
	.venv/bin/ruff check src tests scripts
	.venv/bin/ruff format --check src tests scripts

prompts: ## Build the 40,000-prompt multi-domain pool
	$(RUN) -m fleet.cli build-prompts --total 40000

precision: db ## Greedy-equivalence precision study (fp32 then fp16)
	$(RUN) scripts/precision_study.py --dtype float32
	$(RUN) scripts/precision_study.py --dtype float16

bench-smoke: db ## Fast end-to-end benchmark (minutes, tiny counts)
	FLEET_TRAIN_MIN_SAMPLES=10 FLEET_CANARY_MIN_REQUESTS=6 FLEET_DRIFT_WINDOW=60 \
	$(RUN) -m fleet.bench.run --device cuda --pool-total 1200 --decode-n 64 \
	  --warmup-n 24 --max-new-tokens 32 --pre-n 30 --post-n 50 --oracle-train-n 15 \
	  --window 8 --sustain 5 --periodic-every 25 --audit-every 12 --out runs/smoke

bench: db prompts ## Full benchmark on the 40k pool (hours; GPU)
	$(RUN) -m fleet.bench.run --device cuda --pool-total 40000 \
	  --decode-n 1600 --warmup-n 400 --max-new-tokens 48 \
	  --pre-n 400 --post-n 600 --oracle-train-n 200 \
	  --window 30 --sustain 20 --periodic-every 200 --audit-every 20

bench-report: ## Re-render the most recent benchmark JSON
	$(RUN) -m fleet.bench.report runs/latest.json

fleet-build: ## Build the serving container image
	$(COMPOSE) build

fleet-up: ## Start the whole containerized fleet
	$(COMPOSE) up -d
	@echo "router      http://localhost:8080"
	@echo "replica-a   http://localhost:8081"
	@echo "replica-b   http://localhost:8082"
	@echo "status      make fleet-status"

fleet-down: ## Stop the fleet
	$(COMPOSE) down

fleet-logs: ## Tail fleet logs
	$(COMPOSE) logs -f --tail=80 replica-a replica-b trainer controller

fleet-status: ## Registry state + what each replica is serving
	@curl -fsS http://localhost:8080/fleet/status | $(PY) -m json.tool

fleet-traffic: ## Drive traffic at the fleet (see scripts/drive_traffic.py --help)
	$(RUN) scripts/drive_traffic.py --requests 400

clean: ## Remove run artifacts and caches
	rm -rf runs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

nuke: ## Remove everything including the database volume and prompt cache
	$(COMPOSE) down -v
	rm -rf runs data artifacts/*.json artifacts/.cache
