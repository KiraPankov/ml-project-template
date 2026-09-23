.PHONY: install lint format test train serve docker-build docker-run

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest --cov=mlproject --cov-report=term-missing

train:
	python -m mlproject.train --config configs/default.yaml

serve:
	uvicorn mlproject.api:app --reload

docker-build:
	docker build -t mlproject:latest .

docker-run:
	docker run --rm -p 8000:8000 mlproject:latest
