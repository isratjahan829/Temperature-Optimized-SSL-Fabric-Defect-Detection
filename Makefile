.PHONY: help install install-dev demo test lint format notebook app clean

PYTHON ?= python
SMOKE_CONFIG = configs/smoke_cpu.yaml
SMOKE_EXPERIMENT = outputs/smoke_simclr_resnet18

help:
	@echo "install      install the package"
	@echo "install-dev  install with dev + app extras"
	@echo "demo         synthetic data -> pretrain -> probe -> finetune -> explain"
	@echo "test         run the pytest suite"
	@echo "lint         run ruff"
	@echo "notebook     regenerate and execute the notebook"
	@echo "app          serve the Flask app on the demo checkpoint"
	@echo "clean        remove caches and generated artefacts"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[app,dev]"

demo:
	$(PYTHON) -m fabricssl.cli make-sample-data --root data/sample --images-per-class 40 --image-size 96
	$(PYTHON) -m fabricssl.cli run --config $(SMOKE_CONFIG)
	$(PYTHON) -m fabricssl.cli explain --config $(SMOKE_CONFIG) \
		--checkpoint $(SMOKE_EXPERIMENT)/finetune_best.pt --num-samples 4 --plot

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check src tests app scripts

format:
	$(PYTHON) -m ruff format src tests app scripts

notebook:
	$(PYTHON) scripts/build_notebook.py --execute

app:
	$(PYTHON) app/app.py --checkpoint $(SMOKE_EXPERIMENT)/finetune_best.pt --image-size 64

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/**/__pycache__ .coverage htmlcov build dist *.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
