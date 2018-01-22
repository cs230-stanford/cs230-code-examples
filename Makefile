PYTHON=python3

build:
	cd data/NER; \
	$(PYTHON) build.py;

train:
	$(PYTHON) train.py --model_dir='experiments/test'

evaluate:
	$(PYTHON) evaluate.py --model_dir='experiments/test'

test: build train evaluate