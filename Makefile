.PHONY: test model-recovery-progress recover-models compact-model-progress compact-models gpu-model-validation-dry-run gpu-model-validation gpu-baseline-dry-run gpu-baseline

PYTHON ?= /Users/palm/opt/miniconda3/bin/python3

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

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
