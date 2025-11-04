PY := python3

.PHONY: setup fmt lint test run train

setup:
	$(PY) -m pip install -r requirements.txt
	pre-commit install

fmt:
	black src tests
	isort src tests

lint:
	flake8 src tests

test:
	PYTHONPATH=src pytest -q --maxfail=1

run:
	PYTHONPATH=src $(PY) -m graph_attention_networks.cli

train:
	PYTHONPATH=src $(PY) -c "from graph_attention_networks.train import train; print(train())"
