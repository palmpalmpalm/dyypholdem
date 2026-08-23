.PHONY: test web-install web-test web-build model-recovery-progress recover-models compact-model-progress compact-models gpu-model-validation-dry-run gpu-model-validation gpu-baseline-dry-run gpu-baseline play-ui-dry-run play-ui play-ui-status play-ui-logs play-ui-stop random-benchmark-dry-run random-benchmark

PYTHON ?= /Users/palm/opt/miniconda3/bin/python3
NPM ?= npm

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
