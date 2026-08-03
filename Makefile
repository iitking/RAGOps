.PHONY: help install test index eval gate run docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install    - Install project dependencies"
	@echo "  make test       - Run unit and integration tests"
	@echo "  make index      - Ingest and index raw documents"
	@echo "  make eval       - Run RAG evaluation benchmark"
	@echo "  make gate       - Check CI quality gates against report"
	@echo "  make run        - Start FastAPI server on port 8000"
	@echo "  make docker-up  - Start services via Docker Compose"
	@echo "  make clean      - Clean cache and generated test files"

install:
	pip install -e ".[dev]"

test:
	python3 -m unittest discover tests -v

index:
	python3 -m scripts.build_index --input data/raw

eval:
	python3 -m app.evaluation.run_eval --dataset data/eval/test.jsonl --output reports/eval.json

gate:
	python3 -m app.evaluation.quality_gates --report reports/eval.json

run:
	python3 -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

docker-up:
	docker compose up --build

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov .mypy_cache
