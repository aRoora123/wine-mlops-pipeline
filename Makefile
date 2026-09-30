install:
	pip install --upgrade pip
	pip install -r requirements.txt

lint:
	flake8 src tests --max-line-length=100

test: train
	pytest -v

train:
	python -m src.train

clean:
	find . -name "*.pyc" -delete
	rm -rf __pycache__ .pytest_cache src/__pycache__ tests/__pycache__