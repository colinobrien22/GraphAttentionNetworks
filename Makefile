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
	python3 -m graph_attention_networks.cli train --config configs/default.yaml

train:
	python3 -m graph_attention_networks.cli train --config configs/default.yaml --epochs 50

eval:
	python3 -m graph_attention_networks.cli eval --model models/gat_cora.pt
