# Makefile
.PHONY: install test run lint format clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

run:
	python -m src.main

run-cli:
	python -m src.cli --help

lint:
	ruff check src tests
	mypy src

format:
	black src tests
	ruff format src tests

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf build dist *.egg-info

download-models:
	python scripts/download_models.py