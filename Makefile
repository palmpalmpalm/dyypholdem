.PHONY: test web-install web-test web-build model-recovery-progress recover-models compact-model-progress compact-models gpu-model-validation-dry-run gpu-model-validation gpu-baseline-dry-run gpu-baseline play-ui-dry-run play-ui play-ui-status play-ui-logs play-ui-stop random-benchmark-dry-run random-benchmark solver-regression-preflight solver-regression-river solver-regression-compare

PYTHON ?= /Users/palm/opt/miniconda3/bin/python3
NPM ?= npm
SOLVER_REGRESSION_SOURCE_ROOT ?= .
SOLVER_REGRESSION_ASSET_ROOT ?= .
SOLVER_REGRESSION_MODEL_ROOT ?= runs/model-recovery/compact
SOLVER_REGRESSION_OUTPUT ?= runs/solver-regression/current-river.json
SOLVER_REGRESSION_DEVICE ?= cpu

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

web-install:
	cd web && $(NPM) ci

web-test: web-install
	cd web && $(NPM) test

web-build: web-install
	cd web && $(NPM) run build

model-recovery-progress:
	$(PYTHON) scripts/recover_models.py --progress-report

recover-models:
	$(PYTHON) scripts/recover_models.py

compact-model-progress:
	$(PYTHON) scripts/convert_recovered_models.py --progress-report

compact-models:
	$(PYTHON) scripts/convert_recovered_models.py

gpu-model-validation-dry-run:
	./scripts/run_gpu_baseline.sh --models-dry-run

gpu-model-validation:
	./scripts/run_gpu_baseline.sh --models

gpu-baseline-dry-run:
	./scripts/run_gpu_baseline.sh --dry-run

gpu-baseline:
	./scripts/run_gpu_baseline.sh

play-ui-dry-run:
	./scripts/run_play_ui.sh dry-run

play-ui: web-test web-build
	./scripts/run_play_ui.sh start

play-ui-status:
	./scripts/run_play_ui.sh status

play-ui-logs:
	./scripts/run_play_ui.sh logs

play-ui-stop:
	./scripts/run_play_ui.sh stop

random-benchmark-dry-run: web-test web-build
	DYYPHOLDEM_UI_HANDS=100 DYYPHOLDEM_UI_OPPONENT=random ./scripts/run_play_ui.sh dry-run

random-benchmark: web-test web-build
	DYYPHOLDEM_UI_HANDS=100 DYYPHOLDEM_UI_OPPONENT=random ./scripts/run_play_ui.sh start

solver-regression-preflight:
	$(PYTHON) scripts/solver_regression.py preflight \
		--source-root "$(SOLVER_REGRESSION_SOURCE_ROOT)" \
		--asset-root "$(SOLVER_REGRESSION_ASSET_ROOT)" \
		--model-root "$(SOLVER_REGRESSION_MODEL_ROOT)" \
		--device "$(SOLVER_REGRESSION_DEVICE)" \
		--spot river-7d7c8s5sQd

solver-regression-river:
	$(PYTHON) scripts/solver_regression.py capture \
		--source-root "$(SOLVER_REGRESSION_SOURCE_ROOT)" \
		--asset-root "$(SOLVER_REGRESSION_ASSET_ROOT)" \
		--model-root "$(SOLVER_REGRESSION_MODEL_ROOT)" \
		--device "$(SOLVER_REGRESSION_DEVICE)" \
		--spot river-7d7c8s5sQd \
		--iterations 1000 --warmups 1 --repeats 3 --threads 1 \
		--output "$(SOLVER_REGRESSION_OUTPUT)"

solver-regression-compare:
	@test -n "$(SOLVER_REGRESSION_BASELINE)" || \
		{ echo "set SOLVER_REGRESSION_BASELINE" >&2; exit 2; }
	@test -n "$(SOLVER_REGRESSION_CANDIDATE)" || \
		{ echo "set SOLVER_REGRESSION_CANDIDATE" >&2; exit 2; }
	$(PYTHON) scripts/solver_regression.py compare \
		--baseline "$(SOLVER_REGRESSION_BASELINE)" \
		--candidate "$(SOLVER_REGRESSION_CANDIDATE)"
