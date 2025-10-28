PYTHON ?= python
SHELL := /bin/bash

.PHONY: fmt lint test smoke train-ppo-cartpole train-sac-pendulum offline-cql

fmt:
	ruff check --fix minimalrl configs
	black minimalrl configs

lint:
	ruff check minimalrl configs
	black --check minimalrl configs
	mypy minimalrl

yapf: fmt

test:
	PYTHONPATH=. pytest -n auto minimalrl/tests

smoke: lint
	PYTHONPATH=. pytest -k "advantage or sac_shapes" minimalrl/tests

train-ppo-cartpole:
	bash minimalrl/examples/train_ppo_cartpole.sh

train-sac-pendulum:
	bash minimalrl/examples/train_sac_pendulum.sh

offline-cql:
	bash minimalrl/examples/offline_cql_halfcheetah.sh
