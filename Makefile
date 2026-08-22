.PHONY: test gpu-baseline-dry-run gpu-baseline

PYTHON ?= /Users/palm/opt/miniconda3/bin/python3

test:
	$(PYTHON) tests/test_bucketer_lazy.py

gpu-baseline-dry-run:
	./scripts/run_gpu_baseline.sh --dry-run

gpu-baseline:
	./scripts/run_gpu_baseline.sh
