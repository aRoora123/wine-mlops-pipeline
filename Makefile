install:
	pip install -r requirements.txt

lint:
	flake8 src tests

test:
	pytest -q

train:
	python -m src.train

clean:
	rm -rf __pycache__ .pytest_cache src/__pycache__ tests/__pycache__