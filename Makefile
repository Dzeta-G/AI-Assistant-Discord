.PHONY: run test install

run:
	python -m src.main

test:
	pytest -q

install:
	pip install -e .

