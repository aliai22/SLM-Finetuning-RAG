.PHONY: setup lint fmt test unit train-slm train-emb build-index rag eval

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e ".[dev]"

lint:
	ruff check src

fmt:
	black src && ruff check --fix src && isort src

test:
	pytest -q

# Example task wiring (adapt to your scripts)
train-slm:
	python scripts/train_slm.py --config configs/slm/qa.yaml

train-emb:
	python scripts/finetune_embed.py --config configs/embed/qa.yaml

build-index:
	python scripts/build_index.py --config configs/index/textbooks.yaml

rag:
	python scripts/run_rag.py --config configs/pipeline/v1.yaml

eval:
	python scripts/eval_pipeline.py --config configs/pipeline/v1.yaml
