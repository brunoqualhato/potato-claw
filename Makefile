.PHONY: install dev lint test run api docker

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest --tb=short -q

run:
	python main.py

api:
	pip install fastapi uvicorn && uvicorn api:app --host 0.0.0.0 --port 8000

docker:
	docker build -t potato-claw . && docker run --rm -it potato-claw
